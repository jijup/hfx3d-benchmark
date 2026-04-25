# HFX3D Label Taxonomy

12-class semantic segmentation taxonomy for urban building facades.

| ID | Class | Description | Frequency |
|----|-------|-------------|-----------|
| 0 | wall | Exterior building walls — the dominant class | Very high |
| 1 | window | Glass window panes, frames, and surrounding structure | High |
| 2 | door | Building entrance and exit doors, including frames | Low |
| 3 | balcony | Protruding balcony structures, railings, and floor slabs | Low |
| 4 | vegetation | Trees, shrubs, hedges, and grass near the building | Medium |
| 5 | stairs | Exterior stairways and steps | Rare |
| 6 | terrain | Ground surface — sidewalks, pavement, soil | Medium |
| 7 | roof | Roof surfaces, overhangs, eaves, and gutters | Low |
| 8 | blinds | Window blinds, shutters, and sun-shading elements | Low |
| 9 | other | Miscellaneous facade elements not covered by other classes | Medium |
| 10 | column | Structural columns, pillars, and vertical supports | Rare |
| 11 | arch | Archways, curved openings, and vaulted structures | Rare |

## Notes

- **Class imbalance is severe**: wall and terrain dominate; door, stairs, column, and arch are rare
- All current baseline models score 0% IoU on column and arch
- RGB colour is unreliable (~70% of points have near-black RGB due to the ZEB scanner)
- Models relying heavily on colour should be expected to underperform

## Label Map

```json
{"wall": 0, "window": 1, "door": 2, "balcony": 3, "vegetation": 4,
 "stairs": 5, "terrain": 6, "roof": 7, "blinds": 8, "other": 9,
 "column": 10, "arch": 11}
```
