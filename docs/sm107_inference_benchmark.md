# SM 10.7 grouped inference benchmark

One GPU measures the expert compute assigned to one balanced EP rank. With 2,048 tokens per source rank, 384 global experts and six distinct routes per token, each rank receives 12,288 routed rows. EP sizes 8, 16, 32 and 64 give 48, 24, 12 and 6 local experts with 256, 512, 1,024 and 2,048 rows per expert, respectively.

W13 uses a grouped projection with K = 7168 and N = 2F. W2 uses K = F and N = 7168. The harness starts with packed operands in expert order and writes BF16 projections. The pair measurement replays both projections sequentially using a prepared W2 input. Dispatch, SwiGLU, activation quantization, combine and network transfers are outside the timed region.

## Results

Median CUDA-event latencies in microseconds across five windows of twenty replays. The baseline uses the unmodified kernels from revision `5efe8024099dc9bee03b1af065e2d4b0fd2029d8`. The tuned column uses this SM 10.7 change on the same GPU with the same seeded packed values and nonuniform blocked scales. The GPU has 212 SMs. The runtime was PyTorch 2.15.0a0, CUDA 13.5 and CUTLASS DSL 4.8.0.

| Modeled EP | F | NVFP4 baseline | NVFP4 tuned | Speedup | MXFP8 baseline | MXFP8 tuned | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1536 | 132.1 | 119.7 | 1.10× | 208.8 | 185.4 | 1.13× |
| 8 | 3072 | 217.3 | 196.0 | 1.11× | 369.5 | 325.3 | 1.14× |
| 8 | 6144 | 401.6 | 354.3 | 1.13× | 722.3 | 621.3 | 1.16× |
| 16 | 1536 | 115.0 | 96.4 | 1.19× | 177.4 | 148.9 | 1.19× |
| 16 | 3072 | 182.5 | 150.0 | 1.22× | 304.9 | 246.3 | 1.24× |
| 16 | 6144 | 340.9 | 279.5 | 1.22× | 623.0 | 530.9 | 1.17× |
| 32 | 1536 | 108.4 | 90.2 | 1.20× | 164.7 | 134.4 | 1.23× |
| 32 | 3072 | 172.6 | 140.2 | 1.23× | 284.3 | 234.1 | 1.21× |
| 32 | 6144 | 315.8 | 239.0 | 1.32× | 585.2 | 499.1 | 1.17× |
| 64 | 1536 | 105.3 | 87.8 | 1.20× | 158.4 | 130.4 | 1.21× |
| 64 | 3072 | 168.1 | 136.5 | 1.23× | 267.5 | 229.5 | 1.17× |
| 64 | 6144 | 306.3 | 232.0 | 1.32× | 565.5 | 489.5 | 1.16× |

The [per-projection CSV](../benchmarks/results/sm107/projection_timings.csv) includes medians, window ranges and speedups. The raw records include every window for W13, W2 and the pair, plus both launch configurations. [NVFP4 baseline](../benchmarks/results/sm107/nvfp4-baseline.jsonl), [NVFP4 tuned](../benchmarks/results/sm107/nvfp4-tuned.jsonl), [MXFP8 baseline](../benchmarks/results/sm107/mxfp8-baseline.jsonl), [MXFP8 tuned](../benchmarks/results/sm107/mxfp8-tuned.jsonl).

## Instruction and stage sweep

At EP 16 and F = 3072, the measured pair latencies in microseconds were as follows. The first two rows isolate the change to the NVFP4 N dimension. MXFP8 already uses the full N=256 atom.

| Configuration | NVFP4 | MXFP8 |
| --- | ---: | ---: |
| Legacy instructions, five stages | 182.5 | 304.9 |
| Legacy instructions, full-width N, five stages | 177.3 | 304.9 |
| Wide K, full-width N, five stages with overlapping accumulators | 171.8 | 286.7 |
| Wide K, five stages with independent accumulators | 171.7 | 286.9 |
| Wide K, six stages with independent accumulators | 157.6 | 264.3 |
| Wide K, seven stages with independent accumulators | 153.9 | 257.7 |
| Wide K, eight stages with independent accumulators | 150.0 | 246.3 |

The wide path selects homogeneous NVFP4 or MXFP8 E4M3 inference tiles with one MMA along M, 128 M rows per CTA, and N of 128 or 256. NVFP4 consumes K=128 per instruction and MXFP8 consumes K=64. Both execute two instructions per operand stage at the measured tile K. The SM 10.7 exclusive allocation provides 576 TMEM columns for independent accumulator stages and scales. Fused prefill kernels select up to eight operand stages using their own shared-storage estimates and the 327 KiB oversized SMEM limit with a 1 KiB launch margin. At D = 7168, F = 1536 and 2,017 top-6 tokens across six local experts, staged NVFP4 selected seven stages. Staged MXFP8 and both Mega kernels selected eight. Compact decode has a separate tile policy. [Stage sweep windows](../benchmarks/results/sm107/ablation.jsonl).

Both generated projections contained `UTCOMMA.2CTA.BLOCK16` for NVFP4 and `UTCQMMA.2CTA` for MXFP8. Compute Sanitizer memcheck reported zero errors for both eight-stage projection pairs at modeled EP 64 and F = 1536.

## Reproduction

Run from the repository root on one idle SM 10.7 GPU. Each case checks exact eager/graph agreement and compares sampled outputs from every expert with independently decoded FP32 dot products before timing.

```sh
for fmt in nvfp4 mxfp8; do
  python -m benchmarks.blockscaled_inference --format "$fmt" > "$fmt-baseline.jsonl"
  python -m benchmarks.blockscaled_inference --format "$fmt" \
    --wide --separate --stages 8 > "$fmt-tuned.jsonl"
done
```

Use `--ep 16 --width 3072` to run the representative case. Use `--full-n` for the legacy full-width atom, `--wide` for the five-stage wide atom, and `--separate` to give each accumulator its own TMEM columns.
