# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure one balanced EP rank's MXFP8 backward with prepared CUDA graphs."""

import argparse
import json
import statistics
import tempfile

import torch
import torch.distributed as dist

import dist_moe


def measure(pipeline: str, ep: int, width: int) -> dict:
    """Measure the complete backward after an untimed forward on one real rank."""
    groups = 384 // ep
    tokens, hidden, top_k = 2048, 7168, 6
    config = dist_moe.Config(
        max_num_local_input_tokens=tokens,
        hidden_dim=hidden,
        intermediate_dim=width,
        num_experts=groups,
        top_k=top_k,
        max_moe_layers_per_activation_slot=1,
        num_activation_slots=1,
        activation_slot_capacity_factor=1.0,
        device_scratch_capacity_factor=1.0,
        block_scaled=dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3, pipeline=pipeline
        ),
    )
    context = dist_moe.create_context(group=dist.group.WORLD, config=config)
    try:

        def rand(*shape: int, scale: float = 1.0) -> torch.Tensor:
            """Allocate a differentiable BF16 input."""
            tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
            return tensor.mul_(scale).requires_grad_()

        x_TD = rand(tokens, hidden)
        expert_ids_TK = (
            torch.arange(tokens * top_k, device="cuda").reshape(tokens, top_k) % groups
        )
        scores_TK = torch.full(
            (tokens, top_k), 1 / top_k, device="cuda", requires_grad=True
        )
        w13_E2FD = rand(groups, 2 * width, hidden, scale=hidden**-0.5)
        w2_EDF = rand(groups, hidden, width, scale=width**-0.5)
        inputs = (x_TD, expert_ids_TK, scores_TK, w13_E2FD, w2_EDF)
        grad_TD = torch.randn_like(x_TD)

        def forward(*inputs):
            """Run the public training entry point with the measured policy."""
            return dist_moe.routed_experts(*inputs, context)

        graph = torch.cuda.make_graphed_callables(forward, inputs, num_warmup_iters=2)
        windows = []
        for _ in range(6):
            events = []
            for _ in range(10):
                out = graph(*inputs)
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                torch.autograd.grad(out, (x_TD, scores_TK, w13_E2FD, w2_EDF), grad_TD)
                end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            windows.append(sum(a.elapsed_time(b) for a, b in events) / len(events))
        return dict(
            pipeline=pipeline,
            ep=ep,
            width=width,
            median_ms=statistics.median(windows[1:]),
            windows_ms=windows[1:],
        )
    finally:
        context.close()


def main() -> None:
    """Run one or more local expert geometries on GPU zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ep", type=int, nargs="+", choices=(8, 16, 32, 64), default=[16]
    )
    parser.add_argument(
        "--width", type=int, nargs="+", choices=(1536, 3072, 6144), default=[3072]
    )
    parser.add_argument("--pipeline", choices=("staged", "mega"), required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory() as rendezvous:
        dist.init_process_group(
            "nccl",
            init_method=f"file://{rendezvous}/store",
            rank=0,
            world_size=1,
            device_id=torch.device("cuda:0"),
        )
        try:
            for ep in args.ep:
                for width in args.width:
                    torch.manual_seed(17)
                    print(json.dumps(measure(args.pipeline, ep, width)), flush=True)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
