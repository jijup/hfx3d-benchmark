## Dataset Split

The 26 buildings are divided into an 18 / 4 / 4 train / val / test split assigned at the building level. Split assignments live in `data/splits/{train,val,test}.txt` — edit those files to change the split; the analyser script reads them at runtime.

### Split assignment

| Split | Buildings | Count |
|-------|-----------|------:|
| Train | BLD001, BLD002, BLD003, BLD004, BLD006, BLD007, BLD009, BLD010, BLD011, BLD012, BLD013, BLD014, BLD015, BLD016, BLD018, BLD019, BLD020, BLD023 | 18 |
| Val   | BLD008, BLD022, BLD024, BLD025 | 4 |
| Test  | BLD005, BLD017, BLD021, BLD026 | 4 |

### Point distribution

| Split | Points | Share |
|-------|-------:|------:|
| Train | 343,676,263 | 69.9 % |
| Val   |  68,447,591 | 13.9 % |
| Test  |  79,731,620 | 16.2 % |
| **Total** | **491,855,474** | |

### Per-class split quality

| Class | Total points | Train % | Val % | Test % |
|-------|-------------:|--------:|------:|-------:|
| wall | — | — | — | — |
| window | — | — | — | — |
| door | — | — | — | — |
| balcony | — | — | — | — |
| vegetation | — | — | — | — |
| stairs | — | — | — | — |
| terrain | — | — | — | — |
| roof | — | — | — | — |
| blinds | — | — | — | — |
| other | — | — | — | — |
| column | — | — | — | — |
| arch | — | — | — | — |

> Run `python data/train_test_split_analyzer.py <las_dir>` to regenerate per-class statistics.
