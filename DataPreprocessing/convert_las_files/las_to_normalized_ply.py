import os
import sys
import numpy as np
import argparse
from pathlib import Path


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

        # Extract RGB colors and normalize to 0-1
        rgb = None
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float32)
            # Normalize RGB to 0-1 range
            rgb = rgb / 65535.0

        # Extract intensity and normalize to 0-1
        intensity = None
        if hasattr(las, 'intensity'):
            intensity_raw = np.array(las.intensity).astype(np.float32)
            # Normalize intensity to 0-1 range
            if intensity_raw.max() > intensity_raw.min():
                intensity = (intensity_raw - intensity_raw.min()) / (intensity_raw.max() - intensity_raw.min())
            else:
                intensity = np.zeros_like(intensity_raw)

        # Extract classification
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


def save_ply_format(data, output_path):
    """Save point cloud data in PLY format - all normalized to 0-1 range"""
    # Normalize XYZ to unit sphere
    normalized_xyz = normalize_to_unit_sphere(data['xyz'])
    n_points = len(normalized_xyz)

    with open(output_path, 'w') as f:
        # Write PLY header with all float properties for normalized data
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if data['rgb'] is not None:
            f.write("property float red\n")
            f.write("property float green\n")
            f.write("property float blue\n")

        if data['intensity'] is not None:
            f.write("property float intensity\n")

        if data['labels'] is not None:
            f.write("property uchar label\n")

        f.write("end_header\n")

        # Write data points
        for i in range(n_points):
            line_parts = []

            # XYZ coordinates (normalized to unit sphere)
            line_parts.extend([
                f"{normalized_xyz[i, 0]:.6f}",
                f"{normalized_xyz[i, 1]:.6f}",
                f"{normalized_xyz[i, 2]:.6f}"
            ])

            # RGB colors (if available, normalized to 0-1)
            if data['rgb'] is not None:
                line_parts.extend([
                    f"{data['rgb'][i, 0]:.6f}",
                    f"{data['rgb'][i, 1]:.6f}",
                    f"{data['rgb'][i, 2]:.6f}"
                ])

            # Intensity (if available, normalized to 0-1)
            if data['intensity'] is not None:
                line_parts.append(f"{data['intensity'][i]:.6f}")

            # Classification
            if data['labels'] is not None:
                line_parts.append(str(data['labels'][i]))

            f.write(" ".join(line_parts) + "\n")


def process_directory(input_dir, output_dir):
    """Process all LAS files in input directory and save to PLY format"""
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
            output_file = os.path.join(output_split_dir, f"{las_file.stem}.ply")
            save_ply_format(data, output_file)
            print(f"Saved processed data to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert LAS files to normalized PLY format')
    parser.add_argument('input', help='Input directory containing train/test splits or single LAS file')
    parser.add_argument('-o', '--output', help='Output directory for processed PLY files')

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file mode
        output_path = args.output if args.output else f"{input_path.stem}.ply"
        convert_las_to_ply(input_path, output_path)
    elif input_path.is_dir():
        # Directory mode
        output_dir = args.output if args.output else "processed_ply"
        os.makedirs(output_dir, exist_ok=True)
        process_directory(input_path, output_dir)
    else:
        print(f"Error: Path {input_path} does not exist")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage:")
        print("  Single file: python las_to_ply.py input.las [-o output.ply]")
        print("  Directory: python las_to_ply.py input_dir [-o output_dir]")
        print("Input directory should contain 'train' and 'test' subdirectories with LAS files")
    else:
        main()