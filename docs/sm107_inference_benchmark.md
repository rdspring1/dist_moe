# SM 10.7 grouped inference benchmark

One GPU measures the expert compute assigned to one balanced EP rank. With 2,048 tokens per source rank, 384 global experts and six distinct routes per token, each rank receives 12,288 routed rows. EP sizes 8, 16, 32 and 64 give 48, 24, 12 and 6 local experts with 256, 512, 1,024 and 2,048 rows per expert, respectively.

W13 uses a grouped projection with K = 7168 and N = 2F. W2 uses K = F and N = 7168. The harness starts with packed operands in expert order and writes BF16 projections. The pair measurement replays both projections sequentially using a prepared W2 input. Dispatch, SwiGLU, activation quantization, combine and network transfers are outside the timed region.

## Results

Median CUDA-event latencies in microseconds across five windows of twenty replays. The baseline uses the unmodified kernels from revision `5efe8024099dc9bee03b1af065e2d4b0fd2029d8`. The tuned column uses this SM 10.7 change on the same GPU with the same seeded packed values and nonuniform blocked scales. The GPU has 212 SMs. The runtime was PyTorch 2.15.0a0, CUDA 13.5 and CUTLASS DSL 4.8.0.

| Modeled EP  |  F  |  NVFP4 baseline  |  NVFP4 tuned  |  Speedup |
| ---:  |  ---:  |  ---:  |  ---:  |  ---: |
| 8  |  1536  |  132.1  |  119.7  |  1.10× |
| 8  |  3072  |  217.3  |  196.0  |  1.11× |
| 8  |  6144  |  401.6  |  354.3  |  1.13× |
| 16  |  1536  |  115.0  |  96.4  |  1.19× |
| 16  |  3072  |  182.5  |  150.0  |  1.22× |
| 16  |  6144  |  340.9  |  279.5  |  1.22× |
| 32  |  1536  |  108.4  |  90.2  |  1.20× |
| 32  |  3072  |  172.6  |  140.2  |  1.23× |
| 32  |  6144  |  315.8  |  239.0  |  1.32× |
| 64  |  1536  |  105.3  |  87.8  |  1.20× |
| 64  |  3072  |  168.1  |  136.5  |  1.23× |
| 64  |  6144  |  306.3  |  232.0  |  1.32× |

Run both variants from the repository root.

```sh
python -m benchmarks.blockscaled_inference --format nvfp4
python -m benchmarks.blockscaled_inference --format nvfp4 --wide --separate --stages 8
```
