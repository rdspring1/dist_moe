# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run with torchrun --standalone --nproc-per-node=4 -m benchmarks.mxfp8_dispatch_stages."""

import argparse
import importlib
import json
import os
from statistics import median
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

import dist_moe


def sampled_reference(x_TD, ids_TK, w13_EFD, w2_EDF):
    """Compute one dense reference token per source rank across expert owners."""
    tokens = torch.empty_like(x_TD[:4])
    routes = torch.empty_like(ids_TK[:4])
    dist.all_gather_into_tensor(tokens, x_TD[0:1])
    dist.all_gather_into_tensor(routes, ids_TK[0:1])
    outputs = torch.zeros_like(tokens, dtype=torch.float32)
    for source, experts in enumerate(routes.tolist()):
        for expert in experts:
            owner, local = divmod(expert, 24)
            if owner == dist.get_rank():
                gate, up = F.linear(tokens[source], w13_EFD[local]).float().chunk(2)
                hidden = (F.silu(gate) * up).to(tokens.dtype)
                outputs[source] += F.linear(hidden, w2_EDF[local]).float() / 6
    dist.all_reduce(outputs)
    return outputs[dist.get_rank()]


def measure(graph):
    """Time graph replays in five windows after ten warm-ups."""
    for _ in range(10):
        graph.replay()
    timings = []
    for _ in range(5):
        dist.barrier()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / 30)
    return timings


def profile_dispatch(forward):
    """Collect three eager dispatch kernel durations in microseconds."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(3):
            forward()
        torch.cuda.synchronize()
    samples = [
        event.duration_ns() / 1000
        for event in prof.profiler.kineto_results.events()
        if "grouped_gemm_fprop_dispatch" in event.name()
        and event.device_type() == torch.autograd.DeviceType.CUDA
    ]
    assert len(samples) == 3, samples
    return samples


@torch.no_grad()
def run(width, context, policy, device):
    """Compare W13 stage depths using identical balanced inputs and prepared weights."""
    rank = dist.get_rank()
    torch.manual_seed(2026 + rank)
    x_TD = torch.randn(2048, 7168, device=device, dtype=torch.bfloat16)
    order = torch.randperm(96, device=device)
    ids_TK = order[torch.arange(2048 * 6, device=device).reshape(2048, 6) % 96]
    scores_TK = torch.full((2048, 6), 1 / 6, device=device)
    w13_EFD = x_TD.new_empty(24, 2 * width, 7168).normal_(std=7168**-0.5)
    w2_EDF = x_TD.new_empty(24, 7168, width).normal_(std=width**-0.5)
    reference = sampled_reference(x_TD, ids_TK, w13_EFD, w2_EDF)
    operands = [
        dist_moe.prepare_block_scaled_weight(w, policy, inference=True)
        for w in (w13_EFD, w2_EDF)
    ]
    capacity = context.memory_plan.device_scratch_capacity_rows
    assert capacity >= 512 * 24
    module = importlib.import_module("dist_moe.kernels.dist_blockscaled_grouped_gemm")
    original = module.sm107_prefill_stages
    expected_configs = {}
    saved = None

    def forward():
        """Launch the staged inference layer."""
        return dist_moe.routed_experts(x_TD, ids_TK, scores_TK, *operands, context)

    for depth in (5, 6, 6, 5):
        configs = {}

        def select(config, estimate_bytes, **kwargs):
            """Override only dispatch depth and verify both launch configurations."""
            selected = original(config, estimate_bytes, **kwargs)
            dispatch = (
                kwargs["mode"] == module.DistBlockScaledGroupedGemmKernel.DISPATCH_MODE
            )
            if dispatch:
                assert selected["NUM_SMEM_BUFFERS"] == 5 and selected["USE_SM107"]
                # Deliberately exceed the production budget for the six-stage control.
                selected = dict(selected, NUM_SMEM_BUFFERS=depth)
            comparable = dict(selected)
            if dispatch:
                comparable.pop("NUM_SMEM_BUFFERS")
            assert comparable == expected_configs.setdefault(dispatch, comparable)
            configs["w13" if dispatch else "w2"] = {
                "stages": selected["NUM_SMEM_BUFFERS"],
                "estimated_bytes": estimate_bytes(selected, **kwargs),
                "tile": [selected["BLOCK_SIZE_" + dim] for dim in "MNK"],
                "sm107": selected["USE_SM107"],
            }
            return selected

        with patch.object(module, "sm107_prefill_stages", select):
            eager = forward()
            assert configs.keys() == {"w13", "w2"}, configs
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = forward()
            timings = measure(graph)
            torch.testing.assert_close(output, eager, rtol=0, atol=0)
            if saved is None:
                saved = output.clone()
            torch.testing.assert_close(output, saved, rtol=0, atol=0)
            sample = output[0].float()
            error = ((sample - reference).norm() / reference.norm()).item()
            cosine = F.cosine_similarity(sample, reference, dim=0).item()
            assert error < 0.075 and cosine > 0.9975, (error, cosine)
            dist.barrier()
            kernel = profile_dispatch(forward)
        result = dict(
            rank=rank,
            layer_us=timings,
            dispatch_us=kernel,
            relative_l2=error,
            cosine=cosine,
        )
        ranks = [None] * 4
        dist.all_gather_object(ranks, result)
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        width=width,
                        depth=depth,
                        configs=configs,
                        device_capacity_rows=capacity,
                        ranks=ranks,
                        layer_us=max(median(r["layer_us"]) for r in ranks),
                    )
                ),
                flush=True,
            )
        del graph, output


def main():
    """Initialize four SM 10.7 ranks and benchmark one expert width."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(1536, 3072), default=1536)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        assert dist.get_world_size() == 4 and torch.cuda.get_device_capability() == (
            10,
            7,
        )
        policy = dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3, pipeline="staged"
        )
        config = dist_moe.Config(
            max_num_local_input_tokens=2048,
            hidden_dim=7168,
            intermediate_dim=args.width,
            top_k=6,
            num_experts=96,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=1.0,
            activation_slot_capacity_factor=None,
            num_activation_slots=0,
            inference=True,
            block_scaled=policy,
        )
        context = dist_moe.create_context(
            group=dist.group.WORLD, config=config, device=device
        )
        try:
            run(args.width, context, policy, device)
        finally:
            context.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
