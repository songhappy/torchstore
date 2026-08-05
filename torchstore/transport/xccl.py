# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""XCCL transport for TorchStore on Intel XPU.

Mirrors gloo.py but uses ProcessGroupXCCL and keeps tensors on XPU,
avoiding the two device-host copies that gloo would incur.
"""

import asyncio
import os
import socket
import uuid
from datetime import timedelta
from logging import getLogger
from typing import Any, TYPE_CHECKING

import portpicker
import torch
import torch.distributed as dist
from torch.distributed import PrefixStore, ProcessGroup, Store, TCPStore

from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef


logger = getLogger(__name__)

_store_addrs: dict[str, tuple[str, int, str]] = {}

TORCHSTORE_XCCL_ENABLED = os.environ.get("TORCHSTORE_XCCL_ENABLED", "1") == "1"
TORCHSTORE_XCCL_INIT_TIMEOUT = int(
    os.environ.get("TORCHSTORE_XCCL_INIT_TIMEOUT", "120")
)
# Bulk-transfer wait bound. The PG's own timeout covers rendezvous but does not
# fire on a collective that never completes, so without this a stalled transfer
# blocks its thread forever and the caller reports nothing. 0 disables the bound.
TORCHSTORE_XCCL_TRANSFER_TIMEOUT = int(
    os.environ.get("TORCHSTORE_XCCL_TRANSFER_TIMEOUT", "600")
)


def _wait_with_timeout(work: Any, what: str, store_key: str | None) -> None:
    """Wait on a collective, raising instead of blocking forever if it stalls."""
    if TORCHSTORE_XCCL_TRANSFER_TIMEOUT <= 0:
        work.wait()
        return
    timeout = timedelta(seconds=TORCHSTORE_XCCL_TRANSFER_TIMEOUT)
    try:
        work.wait(timeout)
    except Exception as e:
        raise RuntimeError(
            f"xccl {what} did not complete within "
            f"{TORCHSTORE_XCCL_TRANSFER_TIMEOUT}s (store_key={store_key}). "
            "Raise TORCHSTORE_XCCL_TRANSFER_TIMEOUT if the transfer is merely "
            "slow, or set it to 0 to wait indefinitely."
        ) from e


def xccl_available() -> bool:
    """True iff xccl backend and at least one XPU device are available."""
    if not TORCHSTORE_XCCL_ENABLED:
        return False
    if not hasattr(dist, "is_xccl_available"):
        return False
    try:
        if not dist.is_xccl_available():
            return False
    except Exception:
        return False
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    try:
        return torch.xpu.device_count() > 0
    except Exception:
        return False


def _get_hostname() -> str:
    return socket.getfqdn()


def _xpu_device() -> torch.device:
    """Pick the XPU tile this process should use (LOCAL_RANK or xpu:0)."""
    if "LOCAL_RANK" in os.environ:
        try:
            return torch.device(f"xpu:{int(os.environ['LOCAL_RANK'])}")
        except (TypeError, ValueError):
            pass
    return torch.device("xpu:0")


def _xccl_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device,
    group_name: str,
) -> ProcessGroup:
    """Construct a 2-rank ProcessGroup with XCCL backend bound to ``device``.

    Builds the PG directly from the supplied Store (no global state) so
    each TorchStore transfer gets its own private PG. Options must carry
    global_ranks_in_group and group_name, otherwise oneCCL spins up a
    separate internal KVS and the ranks fail to meet.
    """
    from torch.distributed import ProcessGroupXCCL

    if device.type == "xpu":
        torch.xpu.set_device(device)

    # Use group_name as prefix so both sides of a cross-node PG see the same
    # keys regardless of their local device index.
    prefix = f"xccl/{group_name}/"
    backend_prefix_store = PrefixStore(prefix, store)

    options = ProcessGroupXCCL.Options()
    options.global_ranks_in_group = list(range(world_size))
    options.group_name = group_name
    options._timeout = timeout

    backend_class = ProcessGroupXCCL(backend_prefix_store, rank, world_size, options)

    pg = ProcessGroup(backend_prefix_store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.XCCL)
    pg._register_backend(device, ProcessGroup.BackendType.XCCL, backend_class)
    pg._set_group_name(group_name)
    return pg


class XcclProcessGroupCache(TransportCache):
    def __init__(self) -> None:
        self._process_groups: dict[str, ProcessGroup] = {}

    def put(self, store_key: str, pg: ProcessGroup) -> None:
        self._process_groups[store_key] = pg

    def get(self, store_key: str) -> ProcessGroup:
        return self._process_groups[store_key]

    def clear(self) -> None:
        self._process_groups.clear()


class XcclTransportBuffer(TransportBuffer):
    """Device-resident transport using ProcessGroupXCCL (oneCCL).

    Same handshake/send/recv protocol as GlooTransportBuffer but tensors
    stay on XPU throughout.  When batch mode is active, all tensors in a
    get_batch are concatenated into a single flat buffer and transferred
    with one broadcast -- critical for saturating high-BW fabrics like CXI.
    """

    supports_inplace_resharding = False
    supports_batch_gets = True

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None
        self.master_addr: str | None = None
        self.master_port: int | None = None
        self.store_key: str | None = None
        self.is_object: bool = False
        self.objects: Any = None
        self._tcp_store: TCPStore | None = None
        self._pg_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        # Batch get state: metadata per tensor for split after broadcast
        self._batch_metas: list[tuple[torch.Size, torch.dtype] | str | None] = []

    def requires_handshake(self, requests: list[Request]) -> bool:
        volume_id = self.storage_volume_ref.volume_id
        if volume_id in _store_addrs:
            cached_addr = _store_addrs[volume_id]
            self.master_addr = cached_addr[0]
            self.master_port = cached_addr[1]
            self.store_key = cached_addr[2]
            return False
        return True

    async def _pre_handshake(self) -> None:
        volume_id = self.storage_volume_ref.volume_id
        self.store_key = f"torchstore_xccl_{str(uuid.uuid4())[:8]}"
        self.master_addr = _get_hostname()
        self.master_port = portpicker.pick_unused_port()

        logger.info(
            f"[pid={os.getpid()}] xccl handshake with StorageVolume:[{volume_id}] "
            f"TCPStore at {self.master_addr}:{self.master_port}"
        )

        self._tcp_store = TCPStore(
            host_name=self.master_addr,
            port=self.master_port,
            world_size=2,
            is_master=True,
            timeout=timedelta(seconds=TORCHSTORE_XCCL_INIT_TIMEOUT),
            wait_for_workers=False,
        )

        tcp_store = self._tcp_store
        device = _xpu_device()
        group_name = self.store_key

        def create_pg():
            return _xccl_factory(
                store=tcp_store,
                rank=0,
                world_size=2,
                timeout=timedelta(seconds=TORCHSTORE_XCCL_INIT_TIMEOUT),
                device=device,
                group_name=group_name,
            )

        self._pg_task = asyncio.create_task(asyncio.to_thread(create_pg))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_tcp_store"] = None
        state["_pg_task"] = None
        state["_send_task"] = None
        state["_recv_task"] = None
        state["_batch_metas"] = []
        return state

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[None]:
        """Storage-side: build the PG as rank 1."""
        logger.info(
            f"[pid={os.getpid()}] xccl PG setup at "
            f"{self.master_addr}:{self.master_port}"
        )

        master_addr = self.master_addr
        master_port = self.master_port
        device = _xpu_device()
        group_name = self.store_key

        def create_pg():
            tcp_store = TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=2,
                is_master=False,
                timeout=timedelta(seconds=TORCHSTORE_XCCL_INIT_TIMEOUT),
            )
            return _xccl_factory(
                store=tcp_store,
                rank=1,
                world_size=2,
                timeout=timedelta(seconds=TORCHSTORE_XCCL_INIT_TIMEOUT),
                device=device,
                group_name=group_name,
            )

        pg = await asyncio.to_thread(create_pg)
        ctx.get(XcclProcessGroupCache).put(self.store_key, pg)
        return [None]

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        requests: list[Request],
    ) -> None:
        volume_id = self.storage_volume_ref.volume_id
        pg = await self._pg_task

        _store_addrs[volume_id] = (self.master_addr, self.master_port, self.store_key)
        self.storage_volume_ref.transport_context.get(XcclProcessGroupCache).put(
            self.store_key, pg
        )
        self._tcp_store = None
        self._pg_task = None
        logger.info(f"xccl handshake done with StorageVolume:[{volume_id}]")

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        assert len(requests) == 1
        request = requests[0]

        if request.is_object:
            self.is_object = True
            return
        if request.tensor_val is None:
            return

        tensor = request.tensor_val
        self.shape = tensor.shape
        self.dtype = tensor.dtype

        self._send_task = asyncio.create_task(
            self._send_tensor(tensor, self.storage_volume_ref.transport_context)
        )

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        assert len(entries) == 1
        request, maybe_tensor = entries[0]

        if request.is_object:
            self.is_object = True
            return [request.objects]

        tensor = maybe_tensor
        if tensor is None:
            tensor = torch.empty(self.shape, dtype=self.dtype, device=_xpu_device())

        tensor = await self._receive_tensor(tensor, ctx)
        return [tensor]

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        metas = await self.storage_volume_ref.volume.get_meta.call_one(
            [r.meta_only() for r in requests]
        )
        self._batch_metas = []
        total_bytes = 0
        for request, meta in zip(requests, metas):
            if request.tensor_slice is not None:
                shape = torch.Size(request.tensor_slice.local_shape)
                dtype = meta[1]
                self._batch_metas.append((shape, dtype))
                total_bytes += shape.numel() * torch._utils._element_size(dtype)
            elif isinstance(meta, str) or meta is None:
                self._batch_metas.append(meta)
            else:
                self._batch_metas.append(meta)
                total_bytes += meta[0].numel() * torch._utils._element_size(meta[1])

        if total_bytes == 0:
            self.is_object = True
            return

        flat_buf = torch.empty(total_bytes, dtype=torch.uint8, device=_xpu_device())
        self._recv_task = asyncio.create_task(
            self._receive_tensor(flat_buf, self.storage_volume_ref.transport_context)
        )

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        # Collect all tensor data, skip non-tensor (object) entries
        tensors = []
        total_bytes = 0
        for _, data in entries:
            if not isinstance(data, torch.Tensor):
                continue
            tensors.append(data)
            total_bytes += data.numel() * data.element_size()

        if not tensors:
            self.is_object = True
            if len(entries) == 1:
                self.objects = entries[0][1]
            return

        target = _xpu_device()
        flat_buf = torch.empty(total_bytes, dtype=torch.uint8, device=target)
        offset = 0
        for t in tensors:
            t_dev = t.to(target) if t.device != target else t
            if not t_dev.is_contiguous():
                t_dev = t_dev.contiguous()
            nbytes = t_dev.numel() * t_dev.element_size()
            flat_buf[offset : offset + nbytes].copy_(
                t_dev.view(-1).view(torch.uint8)
            )
            offset += nbytes

        await self._send_tensor(flat_buf, ctx)

    async def _handle_storage_volume_response(
        self, requests: list[Request], transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        if transport_buffer.is_object:
            return [transport_buffer.objects]

        if self._recv_task is not None:
            flat_buf = await self._recv_task
            self._recv_task = None
            if flat_buf is None:
                raise RuntimeError(
                    f"receive_tensor returned None (is_object={self.is_object})"
                )

            results = []
            offset = 0
            for meta in self._batch_metas:
                if isinstance(meta, str) or meta is None:
                    results.append(None)
                    continue
                shape, dtype = meta
                nbytes = shape.numel() * torch._utils._element_size(dtype)
                tensor = flat_buf[offset : offset + nbytes].view(dtype).reshape(shape)
                results.append(tensor)
                offset += nbytes
            self._batch_metas = []
            return results

        raise RuntimeError(f"No recv task available (is_object={self.is_object})")

    async def _receive_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext"
    ) -> torch.Tensor:
        target = _xpu_device()
        if tensor.device != target:
            tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device=target)

        pg = transport_context.get(XcclProcessGroupCache).get(self.store_key)

        def do_recv():
            # Use broadcast instead of p2p recv: oneCCL's OFI transport
            # does not reliably support point-to-point across nodes.
            # Sender is rank 1, receiver is rank 0.
            opts = dist.BroadcastOptions()
            opts.rootRank = 1
            opts.rootTensor = 0
            work = pg.broadcast([tensor], opts)
            _wait_with_timeout(work, "recv broadcast", self.store_key)
            torch.xpu.synchronize(target)

        await asyncio.to_thread(do_recv)
        return tensor

    async def _send_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext"
    ) -> None:
        target = _xpu_device()
        if tensor.device != target:
            tensor = tensor.to(target)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        pg = transport_context.get(XcclProcessGroupCache).get(self.store_key)

        def do_send():
            # Use broadcast instead of p2p send: oneCCL's OFI transport
            # does not reliably support point-to-point across nodes.
            # Sender is rank 1, so broadcast from src=1.
            opts = dist.BroadcastOptions()
            opts.rootRank = 1
            opts.rootTensor = 0
            work = pg.broadcast([tensor], opts)
            _wait_with_timeout(work, "send broadcast", self.store_key)
            torch.xpu.synchronize(target)

        await asyncio.to_thread(do_send)

    async def drop(self) -> None:
        if self._send_task is not None:
            await self._send_task
            self._send_task = None
        self.is_object = False
        self.objects = None
