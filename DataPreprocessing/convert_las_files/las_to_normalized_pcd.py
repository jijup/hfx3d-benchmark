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
            # Normalize RGB to 0-1 range (assuming 16-bit values)
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

def save_pcd_format(data, output_path):
    """Save point cloud data in PCD format - ignoring GPS time"""
    # Normalize XYZ to unit sphere
    normalized_xyz = normalize_to_unit_sphere(data['xyz'])
    n_points = len(normalized_xyz)
    
    # Build field definitions based on available data
    fields = ["x", "y", "z"]
    sizes = [4, 4, 4]  # float32 for coordinates
    types = ["F", "F", "F"]
    counts = [1, 1, 1]
    
    if data['rgb'] is not None:
        fields.extend(["r", "g", "b"])
        sizes.extend([4, 4, 4])  # float32 for normalized RGB
        types.extend(["F", "F", "F"])
        counts.extend([1, 1, 1])
    
    if data['intensity'] is not None:
        fields.append("intensity")
        sizes.append(4)  # float32 for normalized intensity
        types.append("F")
        counts.append(1)
        
    if data['labels'] is not None:
        fields.append("label")
        sizes.append(1)  # uint8
        types.append("U")
        counts.append(1)
    
    with open(output_path, 'w') as f:
        # Write PCD header
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write(f"FIELDS {' '.join(fields)}\n")
        f.write(f"SIZE {' '.join(map(str, sizes))}\n")
        f.write(f"TYPE {' '.join(types)}\n")
        f.write(f"COUNT {' '.join(map(str, counts))}\n")
        f.write(f"WIDTH {n_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n_points}\n")
        f.write("DATA ascii\n")
        
        # Write data points
        for i in range(n_points):
            line_parts = []
            
            # XYZ coordinates (normalized to unit sphere)
            line_parts.extend([
                f"{normalized_xyz[i, 0]:.6f}",
                f"{normalized_xyz[i, 1]:.6f}",
                f"{normalized_xyz[i, 2]:.6f}"
            ])
            
            # RGB colors (normalized to 0-1)
            if data['rgb'] is not None:
                line_parts.extend([
                    f"{data['rgb'][i, 0]:.6f}",
                    f"{data['rgb'][i, 1]:.6f}",
                    f"{data['rgb'][i, 2]:.6f}"
                ])
            
            # Intensity (normalized to 0-1)
            if data['intensity'] is not None:
                line_parts.append(f"{data['intensity'][i]:.6f}")
            
            # Classification (if available)
            if data['labels'] is not None:
                line_parts.append(str(data['labels'][i]))

            f.write(" ".join(line_parts) + "\n")

def convert_las_to_pcd(input_path, output_path=None):
    """Convert LAS file to PCD format"""
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist")
        return False
    
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{input_stem}.pcd")
    
    data = read_las_file(input_path)
    if data is None:
        return False
    
    save_pcd_format(data, output_path)
    print(f"Saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert LAS to normalized PCD format')
    parser.add_argument('input', help='Input LAS file')
    parser.add_argument('-o', '--output', help='Output PCD file')
    
    args = parser.parse_args()
    convert_las_to_pcd(args.input, args.output)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python las_to_normalized_pcd.py input.las [-o output.pcd]")
    else:
        main()