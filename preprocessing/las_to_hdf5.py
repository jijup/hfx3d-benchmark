"""
Convert pre-processed LAS files to HDF5 for the HFX3D benchmark.

Normalization (Official HFX3D Input Protocol):
  XYZ       — mean-centered per building, metric scale preserved
  RGB       — divided by 65535.0  → [0, 1]
  Intensity — global percentile normalization: clip to [p1, p99] from
              the training split, then scale to [0, 1]

Usage:
    python las_to_hdf5.py <input_dir> <output_dir>

input_dir must contain train/ and test/ subdirectories with .las files.
p1/p99 are computed from the training split only and applied to both splits.
Normalization stats are written to <output_dir>/normalization_stats.json.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


def read_las_file(file_path):
    try:
        import laspy
    except ImportError:
        print("Error: laspy not found. Install with: pip install laspy")
        sys.exit(1)

    try:
        las = laspy.read(file_path)
        xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

        rgb = None
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float32) / 65535.0

        intensity = None
        if hasattr(las, 'intensity'):
            intensity = np.array(las.intensity, dtype=np.float32)

        labels = None
        if hasattr(las, 'classification'):
            labels = np.array(las.classification, dtype=np.uint8)

        return {'xyz': xyz, 'rgb': rgb, 'intensity': intensity, 'labels': labels}

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def mean_center_xyz(xyz):
    centroid = xyz.mean(axis=0)
    return (xyz - centroid).astype(np.float32), centroid


def normalize_intensity(intensity, p1, p99):
    clipped = np.clip(intensity, p1, p99)
    span = p99 - p1
    if span > 0:
        return ((clipped - p1) / span).astype(np.float32)
    return np.zeros_like(intensity, dtype=np.float32)


def compute_intensity_percentiles(las_files, sample_per_file=100_000):
    """Estimate global p1/p99 from a random sample of each training file."""
    samples = []
    for f in las_files:
        data = read_las_file(str(f))
        if data is None or data['intensity'] is None:
            continue
        raw = data['intensity']
        if len(raw) > sample_per_file:
            idx = np.random.choice(len(raw), sample_per_file, replace=False)
            raw = raw[idx]
        samples.append(raw)

    if not samples:
        print("Warning: no intensity data found; using identity range [0, 1]")
        return 0.0, 1.0

    all_intensity = np.concatenate(samples)
    return float(np.percentile(all_intensity, 1)), float(np.percentile(all_intensity, 99))


def save_hdf5(data, output_path, split, intensity_p1, intensity_p99):
    try:
        import h5py
    except ImportError:
        print("Error: h5py not found. Install with: pip install h5py")
        sys.exit(1)

    centered_xyz, centroid = mean_center_xyz(data['xyz'])

    with h5py.File(output_path, 'w') as f:
        pc = f.create_group('point_cloud')
        pc.create_dataset('xyz', data=centered_xyz, dtype='float32', compression='gzip')

        if data['rgb'] is not None:
            pc.create_dataset('rgb', data=data['rgb'], dtype='float32', compression='gzip')

        if data['intensity'] is not None:
            norm_i = normalize_intensity(data['intensity'], intensity_p1, intensity_p99)
            pc.create_dataset('intensity', data=norm_i, dtype='float32', compression='gzip')

        lbl = f.create_group('labels')
        if data['labels'] is not None:
            lbl.create_dataset('semantic', data=data['labels'], dtype='uint8', compression='gzip')

        meta = f.create_group('metadata')
        meta.create_dataset('centroid', data=centroid.astype(np.float64))
        meta.create_dataset('num_points', data=len(centered_xyz), dtype='int32')
        meta.create_dataset('split', data=split, dtype='S10')
        meta.attrs['intensity_p1'] = intensity_p1
        meta.attrs['intensity_p99'] = intensity_p99
        meta.attrs['has_rgb'] = data['rgb'] is not None
        meta.attrs['has_intensity'] = data['intensity'] is not None
        meta.attrs['has_labels'] = data['labels'] is not None


def process_directory(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for split in ['train', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Pass 1: compute intensity stats from training data only
    train_files = sorted((input_dir / 'train').glob('*.las'))
    if not train_files:
        print(f"Warning: no .las files in {input_dir / 'train'}")
        intensity_p1, intensity_p99 = 0.0, 1.0
    else:
        print(f"Computing global intensity percentiles from {len(train_files)} training file(s) "
              f"(sampling up to 100k pts/file)...")
        intensity_p1, intensity_p99 = compute_intensity_percentiles(train_files)
        print(f"  p1 = {intensity_p1:.4f},  p99 = {intensity_p99:.4f}")

    # Pass 2: convert all splits using the global stats
    for split in ['train', 'test']:
        split_dir = input_dir / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping")
            continue

        las_files = sorted(split_dir.glob('*.las'))
        print(f"\nConverting {len(las_files)} {split} file(s)...")

        for las_path in las_files:
            print(f"  {las_path.name}")
            data = read_las_file(str(las_path))
            if data is None:
                print(f"    Skipped (read error)")
                continue
            output_path = output_dir / split / f"{las_path.stem}.h5"
            save_hdf5(data, str(output_path), split, intensity_p1, intensity_p99)
            print(f"    → {output_path.name}  ({len(data['xyz']):,} pts)")

    # Persist stats so loaders can verify normalization parameters
    stats_path = output_dir / 'normalization_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({'intensity_p1': intensity_p1, 'intensity_p99': intensity_p99}, f, indent=2)
    print(f"\nNormalization stats saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAS files to HDF5 (Official HFX3D Input Protocol)')
    parser.add_argument('input_dir',
                        help='Directory with train/ and test/ subdirs containing .las files')
    parser.add_argument('output_dir',
                        help='Output directory; train/ and test/ subdirs will be created')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' does not exist")
        sys.exit(1)

    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}\n")
    process_directory(args.input_dir, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
