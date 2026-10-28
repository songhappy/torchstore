# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from torchstore.routing._model import KeyRegistration
from torchstore.routing.plan import Balance, RoutingPlan

from .utils import tensor_slice


def test_collapses_replicated_publishers_to_one_deterministic_source() -> None:
    """Choose one stable source when multiple publishers hold identical data."""
    # Build the same plan twice to verify that source selection is reproducible.
    whole = tensor_slice((0,), (8,), global_shape=(8,))
    entry = {"weight": KeyRegistration(whole, dtype=torch.float32)}
    publishers = {"publisher/0": entry, "publisher/1": entry}
    transfers = []
    for _ in range(2):
        plan = RoutingPlan.for_requester("requester", entry, publishers)
        route = plan.routes.lookup("weight")
        assert route is not None
        [transfer] = route.transfers
        transfers.append(transfer)

    assert transfers[0] == transfers[1]
    assert transfers[0].source in publishers
    assert transfers[0].nbytes == 32


def test_rejects_requester_keys_that_no_publisher_has() -> None:
    """Require every requested key to have at least one publisher."""
    whole = tensor_slice((0,), (8,), global_shape=(8,))
    with pytest.raises(KeyError, match="requester keys have no publisher.*local-only"):
        RoutingPlan.for_requester(
            "requester",
            {"local-only": KeyRegistration(whole, dtype=torch.float32)},
            {"publisher": {"published": KeyRegistration(whole, dtype=torch.float32)}},
        )


def test_publisher_plan_skips_requester_transfer_planning() -> None:
    """A publisher only needs metadata for the slice it serves."""
    publisher_slice = tensor_slice((0,), (2,), global_shape=(4,))
    plan = RoutingPlan.for_publisher(
        "publisher",
        {"w": KeyRegistration(publisher_slice, dtype=torch.float32)},
    )

    assert plan.routes.keys["w"].tensor_slice is publisher_slice
    assert plan.routes.lookup("w") is None


def test_requester_requires_a_publisher() -> None:
    """Require at least one publisher before planning requester transfers."""
    with pytest.raises(ValueError, match="at least one publisher"):
        RoutingPlan.for_requester("requester", {}, {})


def test_rejects_inconsistent_global_shapes() -> None:
    """Reject publisher and requester views of different global tensors."""
    publisher = tensor_slice((0,), (8,), global_shape=(8,))
    requester = tensor_slice((0,), (8,), global_shape=(9,))

    with pytest.raises(ValueError, match="inconsistent global shape"):
        RoutingPlan.for_requester(
            "requester",
            {"w": KeyRegistration(requester, dtype=torch.float32)},
            {"publisher": {"w": KeyRegistration(publisher, dtype=torch.float32)}},
        )


def test_rejects_inconsistent_dtypes_with_the_same_element_size() -> None:
    """Do not accept equal-width wire dtypes as interchangeable."""
    whole = tensor_slice((0,), (8,), global_shape=(8,))

    with pytest.raises(ValueError, match="inconsistent dtype"):
        RoutingPlan.for_requester(
            "requester",
            {"w": KeyRegistration(whole, dtype=torch.bfloat16)},
            {"publisher": {"w": KeyRegistration(whole, dtype=torch.float16)}},
        )


@pytest.mark.parametrize(
    "publisher_slices",
    [
        [((0,), (3,))],
        [((0,), (3,)), ((2,), (2,))],
    ],
    ids=("gap", "overlap"),
)
def test_rejects_inexact_publisher_coverage(publisher_slices) -> None:
    """Reject gaps and overlaps that could cause concurrent destination writes."""
    whole = tensor_slice((0,), (4,), global_shape=(4,))
    publishers = {
        f"publisher/{index}": {
            "w": KeyRegistration(
                tensor_slice(offset, shape, global_shape=(4,)),
                dtype=torch.float32,
            )
        }
        for index, (offset, shape) in enumerate(publisher_slices)
    }

    with pytest.raises(ValueError, match="does not cover .* exactly"):
        RoutingPlan.for_requester(
            "requester",
            {"w": KeyRegistration(whole, dtype=torch.float32)},
            publishers,
        )


def test_accepts_a_zero_sized_requester_shard() -> None:
    """Plan an empty destination without manufacturing a data transfer."""
    published = tensor_slice((0,), (2,), global_shape=(2,))
    empty = tensor_slice((2,), (0,), global_shape=(2,))
    plan = RoutingPlan.for_requester(
        "requester",
        {"w": KeyRegistration(empty, dtype=torch.float32)},
        {"publisher": {"w": KeyRegistration(published, dtype=torch.float32)}},
    )

    route = plan.routes.lookup("w")
    assert route is not None
    assert route.destination_slice is empty
    assert route.transfers == ()


def test_requester_balance_reads_every_key_from_one_replica() -> None:
    """Under REQUESTER each requester reaches one replica's volumes, and the
    requesters split evenly over the replicas."""
    num_replicas, num_shards, num_requesters = 8, 12, 96
    rows = 2 * num_shards
    publishers = {}
    for replica in range(num_replicas):
        for shard in range(num_shards):
            shard_slice = tensor_slice((2 * shard, 0), (2, 6), global_shape=(rows, 6))
            publishers[f"publisher/{replica * num_shards + shard}"] = {
                key: KeyRegistration(shard_slice, dtype=torch.float32)
                for key in ("wq", "wk", "wo")
            }
    whole = tensor_slice((0, 0), (rows, 6), global_shape=(rows, 6))
    wanted = {
        key: KeyRegistration(whole, dtype=torch.float32) for key in ("wq", "wk", "wo")
    }

    requesters_per_volume: dict[str, int] = {}
    for requester in range(num_requesters):
        plan = RoutingPlan.for_requester(
            f"requester/{requester}/0", wanted, publishers, balance=Balance.REQUESTER
        )
        sources = {
            transfer.source
            for key in wanted
            for transfer in plan.routes.lookup(key).transfers
        }
        replica = requester % num_replicas
        assert sources == {
            f"publisher/{replica * num_shards + shard}" for shard in range(num_shards)
        }
        for source in sources:
            requesters_per_volume[source] = requesters_per_volume.get(source, 0) + 1

    assert set(requesters_per_volume.values()) == {num_requesters // num_replicas}


def test_rotate_balance_spreads_one_requester_over_replicas() -> None:
    """ROTATE (the default) picks by key, so one requester reaches several replicas."""
    publishers = {
        f"publisher/{replica}": {
            key: KeyRegistration(
                tensor_slice((0,), (8,), global_shape=(8,)), dtype=torch.float32
            )
            for key in ("wq", "wk", "wv", "wo")
        }
        for replica in range(4)
    }
    wanted = {
        key: KeyRegistration(
            tensor_slice((0,), (8,), global_shape=(8,)), dtype=torch.float32
        )
        for key in ("wq", "wk", "wv", "wo")
    }
    plan = RoutingPlan.for_requester("requester/0/0", wanted, publishers)
    sources = {plan.routes.lookup(key).transfers[0].source for key in wanted}
    assert len(sources) > 1
