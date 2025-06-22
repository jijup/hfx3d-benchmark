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
            labels = np.array(las.classification).astype(np.float64)
        
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
    
    return xyz_normalized

def save_xyz_format(data, output_path):
    """Save point cloud data in XYZ format - ignoring GPS time"""
    # Normalize XYZ to unit sphere
    normalized_xyz = normalize_to_unit_sphere(data['xyz'])
    
    with open(output_path, 'w') as f:
        # Header comment
        f.write("//X Y Z R G B Intensity Classification\n")
        
        # Write data points
        for i in range(len(normalized_xyz)):
            line_parts = []
            
            # XYZ coordinates (normalized to unit sphere)
            line_parts.extend([
                f"{normalized_xyz[i, 0]:.8f}",
                f"{normalized_xyz[i, 1]:.8f}",
                f"{normalized_xyz[i, 2]:.8f}"
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
            
            # Classification
            if data['labels'] is not None:
                line_parts.append(f"{data['labels'][i]:.0f}")

            f.write(" ".join(line_parts) + "\n")

def convert_las_to_xyz(input_path, output_path=None):
    """Convert LAS file to XYZ format"""
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist")
        return False
    
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{input_stem}.xyz")
    
    data = read_las_file(input_path)
    if data is None:
        return False
    
    save_xyz_format(data, output_path)
    print(f"Saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert LAS to XYZ format')
    parser.add_argument('input', help='Input LAS file')
    parser.add_argument('-o', '--output', help='Output XYZ file')
    
    args = parser.parse_args()
    convert_las_to_xyz(args.input, args.output)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python las_to_normalized_xyz.py input.las [-o output.xyz]")
    else:
        main()