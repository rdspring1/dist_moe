# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-GPU grouped projection timings for a balanced synthetic EP rank."""

import argparse
import importlib.metadata
import json
import statistics
import subprocess

import torch

from dist_moe.kernels.blockscaled_grouped_gemm import (
    auto_blockscaled_config,
    blockscaled_grouped_gemm_fprop,
    MXFP8_E4M3,
    NVFP4,
)
from dist_moe.kernels.params import _torch_dtype


def projection(fmt, groups, n, k, sms, wide, stages, separate, full_n=False):
    """Capture one grouped projection with fixed, already-dispatched rows."""
    rows = 2048 * 6
    split = torch.full((groups,), rows // groups, dtype=torch.int32, device="cuda")
    packed_k = k // 2 if fmt == NVFP4 else k
    dtype = _torch_dtype(fmt.a_dtype)
    scale_dtype = _torch_dtype(fmt.sf_dtype)
    generator = torch.Generator(device="cuda").manual_seed(17)

    def operand(shape):
        """Generate finite packed values directly on device."""
        bits = torch.randint(
            256, shape, dtype=torch.uint8, device="cuda", generator=generator
        )
        if fmt == MXFP8_E4M3:
            bits.bitwise_and_(0x8F).bitwise_or_(0x30)
        return bits.view(dtype)

    def scales(count):
        """Draw positive scales in the native blocked storage order."""
        low, high = (48, 65) if fmt == NVFP4 else (126, 129)
        return torch.randint(
            low, high, (count,), dtype=torch.uint8, device="cuda", generator=generator
        ).view(scale_dtype)

    x = operand((rows, packed_k))
    w = operand((groups, n, packed_k))
    cols = k // fmt.sf_vec_size
    sa, sb = scales(rows * cols), scales(groups * n * cols)
    y = torch.empty((rows, n), device="cuda", dtype=torch.bfloat16)
    config = auto_blockscaled_config(
        GM=rows,
        G=groups,
        N=n,
        K=k,
        num_sms=sms,
        format=fmt,
        m_multiple_of=rows // groups,
    )

    if wide:
        config["USE_SM107"] = True
    if wide or full_n:
        config["MMA_ATOM_N"] = config["BLOCK_SIZE_N"]
    if separate:
        config["OVERLAPPING_ACCUM"] = False
    if stages:
        config["NUM_SMEM_BUFFERS"] = stages

    def run():
        """Launch with stable operand and output addresses."""
        blockscaled_grouped_gemm_fprop(
            x,
            w,
            sa,
            sb,
            split,
            y=y,
            format=fmt,
            m_multiple_of=rows // groups,
            config=config,
        )

    run()
    eager = y.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(y, eager, rtol=0, atol=0)
    # Check sampled rows and columns from every expert against unpacked FP32 dots.
    group = torch.arange(groups, device="cuda").repeat_interleave(8)
    row = group * (rows // groups) + torch.randint(
        rows // groups, group.shape, device="cuda", generator=generator
    )
    col = torch.randint(n, group.shape, device="cuda", generator=generator)

    def unpack(data, scale, indices):
        """Decode sampled K-major vectors and their blocked scale rows."""
        values = data.view(torch.uint8).reshape(-1, packed_k)[indices]
        if fmt == NVFP4:
            nibble = torch.stack((values & 15, values >> 4), dim=-1).flatten(1)
            levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
            decoded = levels[(nibble & 7).long()] * torch.where(nibble < 8, 1, -1)
        else:
            decoded = values.view(dtype).float()
        c = torch.arange(cols, device="cuda")
        r = indices[:, None]
        offsets = (
            r // 128 * cols * 128
            + c // 4 * 512
            + r % 32 * 16
            + r % 128 // 32 * 4
            + c % 4
        )
        factors = scale.view(torch.uint8)[offsets].view(scale_dtype).float()
        return decoded * factors.repeat_interleave(fmt.sf_vec_size, dim=1)

    expected = (unpack(x, sa, row) * unpack(w, sb, group * n + col)).sum(1)
    torch.testing.assert_close(y[row, col].float(), expected, rtol=0.008, atol=0.03)
    return graph, (x, w, sa, sb, split, y), config


def measure(graphs):
    """Return per-call milliseconds across five warmed replay windows."""
    for graph in graphs:
        graph.replay()
    windows = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            for graph in graphs:
                graph.replay()
        end.record()
        end.synchronize()
        windows.append(start.elapsed_time(end) / 20)
    return dict(median_ms=statistics.median(windows), windows_ms=windows)


def main():
    """Emit a JSON record per format, local expert count, and intermediate width."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("nvfp4",), required=True)
    parser.add_argument(
        "--ep", choices=(8, 16, 32, 64), type=int, nargs="+", default=(8, 16, 32, 64)
    )
    parser.add_argument(
        "--width",
        choices=(1536, 3072, 6144),
        type=int,
        nargs="+",
        default=(1536, 3072, 6144),
    )
    parser.add_argument(
        "--wide", action="store_true", help="use SM107 double-K instructions"
    )
    parser.add_argument(
        "--full-n",
        action="store_true",
        help="use a full-width N atom for the instruction comparison",
    )
    parser.add_argument("--stages", type=int, choices=(5, 6, 7, 8))
    parser.add_argument("--separate", action="store_true")
    args = parser.parse_args()
    if (args.separate or args.stages) and not args.wide:
        parser.error("--separate and --stages require --wide")
    if args.wide and torch.cuda.get_device_capability() != (10, 7):
        parser.error("--wide requires SM 10.7")
    props = torch.cuda.get_device_properties(0)
    fmt = NVFP4
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for ep in args.ep:
        for width in args.width:
            projections = [
                projection(
                    fmt,
                    384 // ep,
                    n,
                    k,
                    props.multi_processor_count,
                    args.wide,
                    args.stages,
                    args.separate,
                    args.full_n,
                )
                for n, k in ((2 * width, 7168), (7168, width))
            ]
            graphs = [p[0] for p in projections]
            print(
                json.dumps(
                    dict(
                        format=args.format,
                        ep=ep,
                        width=width,
                        revision=revision,
                        gpu=props.name,
                        sm_count=props.multi_processor_count,
                        torch_version=torch.__version__.split("+")[0],
                        capability=[props.major, props.minor],
                        cuda=torch.version.cuda,
                        cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
                        configs=[p[2] for p in projections],
                        w13=measure(graphs[:1]),
                        w2=measure(graphs[1:]),
                        pair=measure(graphs),
                    )
                ),
                flush=True,
            )
            del graphs, projections


if __name__ == "__main__":
    main()
