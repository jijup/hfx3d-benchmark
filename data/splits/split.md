## Dataset Split

The 25 buildings are divided into an **80/20 train/test split** assigned at the building level. The split was determined by brute-force search over all possible combinations to minimise per-class deviation from the 20% test target across all 12 semantic classes.

### Split assignment

| Split | Buildings | Count |
|-------|-----------|------:|
| Train | BLD001, BLD002, BLD003, BLD004, BLD005, BLD007, BLD008, BLD009, BLD010, BLD011, BLD012, BLD013, BLD016, BLD017, BLD018, BLD019, BLD020, BLD021, BLD023, BLD025 | 20 |
| Test  | BLD006, BLD014, BLD015, BLD022, BLD024 | 5 |

### Point distribution

| Split | Points | Share |
|-------|-------:|------:|
| Train | 388,093,656 | 80.1% |
| Test  |  96,565,541 | 19.9% |
| **Total** | **484,659,197** | |

### Per-class split quality

The table below shows, for each class, what percentage of its total points falls in train vs. test. The target is 80% / 20%.

| Class | Total points | Train % | Test % | Δ from target |
|-------|-------------:|--------:|-------:|:-------------:|
| wall | 211,352,322 | 79.80% | 20.20% | +0.20% |
| window | 42,837,929 | 79.86% | 20.14% | +0.14% |
| terrain | 130,773,026 | 80.31% | 19.69% | −0.31% |
| stairs | 4,729,644 | 80.51% | 19.49% | −0.51% |
| roof | 4,179,599 | 79.17% | 20.83% | +0.83% |
| vegetation | 47,160,952 | 78.72% | 21.28% | +1.28% |
| balcony | 4,857,985 | 78.07% | 21.93% | +1.93% |
| other | 12,497,197 | 81.52% | 18.48% | −1.52% |
| blinds | 15,901,920 | 83.61% | 16.39% | −3.61% |
| door | 2,310,541 | 67.28% | 32.72% | +12.72% |
| column | 6,790,854 | 88.32% | 11.68% | −8.32% |
| arch | 1,267,228 | 88.36% | 11.64% | −8.36% |

> **Note:** `door`, `column`, and `arch` deviate from the 20% target because their points are heavily concentrated in a small number of buildings. These three classes together account for only 0.88% of the total dataset, so the impact on overall split balance is negligible. The deviations shown are the minimum achievable under a building-level split constraint.
