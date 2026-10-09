# Preliminary comparison: R routine against the port, first 5 participants

Reduced settings for all three builds: 100 starting points (one matrix shared by the participants, seed 1 in R),
radius 1, population 1,000,000, 200 iterations, seed 2026. The R routine ran the participants one after another
with the compiled C++ OSCAR (`compare.R`, psp 1.0.6); the port ran them in lockstep with `run_oscar.py --first 5
--starts 100 --iterations 200 --seed 2026 --no-save`. Machine: 12 CPU threads, NVIDIA T400 (4 GB).

| Participant | R, psp 1.0.6 + C++ | torch GPU float32 | torch CPU float64 |
| --- | --- | --- | --- |
| 615f3150131744464b58c90e | 214 patterns (36,634 evaluations) | 214 (36,282) | 202 (35,626) |
| 61409bc0e94c55a3875d3b92 | 216 (35,926) | 212 (35,977) | 217 (35,944) |
| 6153fef41e57781211baea3a | 201 (35,780) | 204 (34,938) | 203 (36,211) |
| 6158ba08f68a4b368602a5d8 | 203 (35,225) | 194 (34,332) | 202 (36,164) |
| 615635a1c82758c22c5fe5f7 | 201 (35,234) | 205 (36,196) | 215 (36,997) |

Per participant, R and the GPU port share 191 to 207 inequality matrices; the 3 to 12 found by only one of them
hold one or a few evaluations each, so all but a rounding error of the evaluations of either run fall in matrices
both found. The model draws its initial state at random on every evaluation, so no two runs take the same path.

| | Per participant | 5 participants |
| --- | --- | --- |
| R, one participant after another | 240 s | 20.0 min |
| torch CPU float64, lockstep | | 7.3 min (2.7×) |
| torch GPU float32, lockstep | | 1.7 min (12×) |

Files: `r_summary.csv` and `r_patterns.csv` (R: time, counts, every matrix's upper triangle in column-major order
and its population), `torch_*_tracking.csv` (the port's `tracking.csv` files), `compare.R` (the R side).
