# UAV Foreground Generation Benchmark
## Hardware
- CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz * 2
- RAM: Samsung 64GB DDR4-3200 RDIMM ECC Registered * 6
- GPU: NVIDIA GeForce RTX 3090 * 8

## Environment
- Linux kernel: 6.8.0-138-generic
- Host OS: Ubuntu 22.04.1
- Rootless Container OS: Debian bookworm
- OpenMPI: 4.1.4
- Blender: 3.5.1

## Config
[robust_anti_uav_demo.py](../configs/robust_anti_uav_demo.py)

1,000 images = 500 clear + 500 cloudy

## Result (3 runs per setting)
[benchmark.csv](./benchmark.csv)

<!-- | GPUs | Execution Time (avg) [s] | Speedup |
|------|--------------------------|---------|
| 1    | 59880.34                 | 1x      |
| 2    | 29482.83                 | 2.03x   |
| 4    | 7475.32                  | 8.01x*  |
| 8    | 4349.16                  | 13.77x* | -->

| GPUs | Execution Time (avg) [s] | Speedup | Speedup (rel) |
|------|--------------------------|---------|---------------|
| 1    | 59880.34                 | 1x      | 1x            |
| 2    | 29482.83                 | 2.03x   | 2.03x         |
| 4    | 7475.32                  | 8.01x*  | 3.94x         |
| 8    | 4349.16                  | 13.77x* | 1.71x         |

*When we use more than four GPUs, the speedup exceeds linear scaling (Superlinear Speedup).

Why? Cache in VRAM? Memory localization? TODO: need further analysis.
