"""Apply the HFX3D protocol fixes to a Pointcept checkout and generate the
PointGroup config from the base config. Documents every source change; each
replacement must match exactly once, otherwise nothing is written.
Usage: python pg_fresh_patch.py <path-to-Pointcept>."""
import io
import sys

ROOT = sys.argv[1]
FRESH_DATA = "/scratch/abdiaziz/hfx3d/fresh/data/HFX3D_Instance"

edits = {
    # T1: HDF5 colour is [0, 1]; Pointcept's colour transforms (ChromaticAutoContrast/
    # Translation/Jitter, NormalizeColor) assume 0-255. Scale at load so every
    # transform behaves as designed and the model sees [0, 1] colour in train/val/test.
    "pointcept/datasets/hfx3d_inst.py": [
        ('            "color": rgb,\n',
         '            # HDF5 rgb is [0, 1]; Pointcept colour transforms and\n'
         '            # NormalizeColor assume 0-255, so scale here (NormalizeColor maps back to [0, 1]).\n'
         '            "color": rgb * 255.0,\n'),
    ],
    # T2: thing points whose instance is -1 (GT objects < 200 points) must be ignored
    # in instance targets, not merged into one fake instance per crop.
    "pointcept/datasets/transform.py": [
        ("        mask = ~np.in1d(segment, self.segment_ignore_index)\n"
         "        # mapping ignored instance to ignore index\n",
         "        # also ignore thing points whose instance is -1\n"
         "        mask = ~np.isin(segment, self.segment_ignore_index) & (\n"
         "            instance != self.instance_ignore_index\n"
         "        )\n"
         "        # mapping ignored instance to ignore index\n"),
    ],
    # Deterministic test-time voxel sampling: GridSample(mode="train") picks a random
    # point per voxel, so seed it per tile (same checkpoint -> same score).
    "tools/test_fullscene_insseg_hfx3d_fullres.py": [
        ("    for center in centers:\n        idx_full = tile_point_indices(",
         "    for tile_i, center in enumerate(centers):\n"
         "        np.random.seed((seed * 100003 + tile_i) % (2 ** 32))  # deterministic GridSample per tile\n"
         "        idx_full = tile_point_indices("),
    ],
}

config_src = "configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-base.py"
config_dst = "configs/hfx3d/insseg-pointgroup-v1m1-0-spunet-fresh.py"
config_edits = [
    ("num_worker = 1 ", "num_worker = 4  # one loader worker per GPU (4 GPUs)\n# (was 1 for a single GPU) "),
    ('data_root = "/scratch/abdiaziz/hfx3d/HFX3D_Instance"', f'data_root = "{FRESH_DATA}"'),
    ('dict(type="CheckpointSaver", save_freq=4)', 'dict(type="CheckpointSaver", save_freq=20)  # every 1,800 steps'),
]

out, ok = {}, True
for rel, reps in edits.items():
    s = io.open(f"{ROOT}/{rel}", encoding="utf-8").read()
    for old, new in reps:
        if s.count(old) != 1:
            print(f"MISMATCH {rel}: {s.count(old)} matches for {old[:60]!r}")
            ok = False
        s = s.replace(old, new)
    out[rel] = s
cfg = io.open(f"{ROOT}/{config_src}", encoding="utf-8").read()
for old, new in config_edits:
    if cfg.count(old) != 1:
        print(f"MISMATCH config: {cfg.count(old)} matches for {old[:60]!r}")
        ok = False
    cfg = cfg.replace(old, new)
cfg = ("# HFX3D PointGroup config (generated from the base config by pg_fresh_patch.py).\n"
       "# Differences from base: num_worker 1->4 (4 GPUs), data root, checkpoint every\n"
       "# 20 epochs (1,800 steps). Exact budget: 200 epochs x 90 iterations =\n"
       "# 18,000 steps, batch 4. Code fixes in hfx3d_inst.py (colour x255) and\n"
       "# transform.py (ignore instance -1).\n") + cfg
out[config_dst] = cfg
if not ok:
    sys.exit(1)
for rel, s in out.items():
    io.open(f"{ROOT}/{rel}", "w", encoding="utf-8").write(s)
print("patched:", ", ".join(out))
