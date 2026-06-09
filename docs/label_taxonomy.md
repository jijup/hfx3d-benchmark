# HFX3D Label Taxonomy

12-class semantic segmentation taxonomy for urban building facades.

| ID | Class | Description | Frequency |
|----|-------|-------------|-----------|
| 0 | wall | Exterior facade surfaces visible from outside; excludes openings | Very high |
| 1 | window | Openings in a wall or roof for light and ventilation; includes glazing, frames, and any transparent or translucent glass surface on the facade | High |
| 2 | door | Openings for access or egress; includes door panels and frames | Low |
| 3 | balcony | Protruding slabs, floors, and railings; excludes supporting columns | Low |
| 4 | vegetation | Trees, shrubs, hedges, and grass in the building vicinity | Medium |
| 5 | stairs | Exterior steps and stairways; excludes support poles and handrails | Rare |
| 6 | terrain | Ground surfaces outside a building — sidewalks, pavement, soil | Medium |
| 7 | roof | Roof surfaces and associated structures, including dormers | Low |
| 8 | blinds | Window closures (blinds, shutters) — labelled whether open or closed | Low |
| 9 | other | Building-related elements not covered above (e.g. bins, poles) | Medium |
| 10 | column | Structural vertical supports — columns, pillars, and balcony supports | Rare |
| 11 | arch | Downward-facing building surfaces and attached elements (e.g. cameras, lights) | Rare |

## Label Map

See [`configs/label_map.json`](../configs/label_map.json) for the canonical class-name → ID mapping.
