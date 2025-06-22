import os
import sys
import numpy as np
import argparse
from pathlib import Path
import shutil


def read_las_file(file_path):
    """Read LAS file and extract all available data"""
    try:
        import laspy
    except ImportError:
        print("Error: laspy library not found. Please install it with: pip install laspy")
        sys.exit(1)

    try:
        las = laspy.read(file_path)
        print(f"Reading LAS file: {file_path}")
        print(f"Total points: {len(las.points):,}")

        # Extract XYZ coordinates
        xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

        # Extract RGB colors (if available) and normalize to 0-1
        rgb = None
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float32)
            # Normalize RGB to 0-1 range (LAS stores as 16-bit)
            rgb = rgb / 65535.0

        # Extract intensity (if available) and normalize to 0-1
        intensity = None
        if hasattr(las, 'intensity'):
            intensity_raw = np.array(las.intensity).astype(np.float32)
            # Normalize intensity to 0-1 range
            if intensity_raw.max() > intensity_raw.min():
                intensity = (intensity_raw - intensity_raw.min()) / (intensity_raw.max() - intensity_raw.min())
            else:
                intensity = np.zeros_like(intensity_raw)

        # Extract classification (if available)
        labels = None
        if hasattr(las, 'classification'):
            labels = np.array(las.classification).astype(np.uint8)

        return {
            'xyz': xyz,
            'rgb': rgb,
            'intensity': intensity,
            'labels': labels
        }

    except Exception as e:
        print(f"Error reading LAS file {file_path}: {e}")
        return None


def normalize_to_unit_sphere(xyz):
    """Normalize XYZ coordinates to unit sphere (-1 to 1 range)"""
    # Center the points at origin
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid

    # Find the maximum distance from center
    max_distance = np.max(np.linalg.norm(xyz_centered, axis=1))

    # Scale to unit sphere if max_distance > 0
    if max_distance > 0:
        xyz_normalized = xyz_centered / max_distance
    else:
        xyz_normalized = xyz_centered

    print(f"XYZ normalization - Original range: [{xyz.min():.3f}, {xyz.max():.3f}]")
    print(f"XYZ normalization - Normalized range: [{xyz_normalized.min():.3f}, {xyz_normalized.max():.3f}]")

    return xyz_normalized.astype(np.float32)


def save_hdf5_format(data, output_path, split):
    """Save point cloud data in structured HDF5 format"""
    try:
        import h5py
    except ImportError:
        print("Error: h5py library not found. Please install it with: pip install h5py")
        sys.exit(1)

    # Normalize XYZ to unit sphere (-1 to 1)
    normalized_xyz = normalize_to_unit_sphere(data['xyz'])

    with h5py.File(output_path, 'w') as f:
        # Point cloud data group
        pc_group = f.create_group('point_cloud')
        pc_group.create_dataset('xyz', data=normalized_xyz, dtype='float32', compression='gzip')

        if data['rgb'] is not None:
            pc_group.create_dataset('rgb', data=data['rgb'], dtype='float32', compression='gzip')

        if data['intensity'] is not None:
            pc_group.create_dataset('intensity', data=data['intensity'], dtype='float32', compression='gzip')

        # Labels group
        label_group = f.create_group('labels')
        if data['labels'] is not None:
            label_group.create_dataset('semantic', data=data['labels'], dtype='uint8', compression='gzip')

        # Metadata group
        meta_group = f.create_group('metadata')
        meta_group.create_dataset('num_points', data=len(normalized_xyz), dtype='int32')
        meta_group.create_dataset('split', data=split, dtype='S10')  # String

        # Add additional metadata as attributes
        meta_group.attrs['has_rgb'] = data['rgb'] is not None
        meta_group.attrs['has_intensity'] = data['intensity'] is not None
        meta_group.attrs['has_labels'] = data['labels'] is not None


def process_directory(input_dir, output_dir):
    """Process all LAS files in input directory and save to structured HDF5 format"""
    # Create output directory structure
    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Process each split
    for split in ['train', 'test']:
        split_dir = os.path.join(input_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split} directory not found: {split_dir}")
            continue

        output_split_dir = train_output_dir if split == 'train' else test_output_dir

        # Process each LAS file in the split directory
        for las_file in Path(split_dir).glob('*.las'):
            print(f"\nProcessing {split} file: {las_file}")
            data = read_las_file(str(las_file))
            if data is None:
                continue

            # Create output filename
            output_file = os.path.join(output_split_dir, f"{las_file.stem}.h5")
            save_hdf5_format(data, output_file, split)
            print(f"Saved processed data to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAS files to structured HDF5 format for semantic segmentation')
    parser.add_argument('input_dir', help='Input directory containing train/test splits')
    parser.add_argument('output_dir', help='Output directory for processed HDF5 files')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    print(f"Processing LAS files from: {args.input_dir}")
    print(f"Saving processed HDF5 files to: {args.output_dir}")

    process_directory(args.input_dir, args.output_dir)
    print("\nProcessing complete!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python process_las_to_hdf5.py input_dir output_dir")
        print("Input directory should contain 'train' and 'test' subdirectories with LAS files")
    else:
        main()