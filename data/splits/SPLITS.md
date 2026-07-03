# Dataset Split

The 26 buildings are split at the building level: 18 train / 4 val / 4 test. Assignments live in `data/splits/{train,val,test}.txt`.

| Split | Buildings | Count |
|-------|-----------|------:|
| Train | BLD001–004, BLD006–007, BLD009–016, BLD018–020, BLD023 | 18 |
| Val   | BLD008, BLD022, BLD024, BLD025 | 4 |
| Test  | BLD005, BLD017, BLD021, BLD026 | 4 |

| Split | Points | Share |
|-------|-------:|------:|
| Train | 343,676,263 | 69.9 % |
| Val   |  68,447,591 | 13.9 % |
| Test  |  79,731,620 | 16.2 % |
| **Total** | **491,855,474** | |

To generate per-class split statistics, run `python preprocessing/analyse_dataset.py <las_dir>`.
