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

def save_hdf5_format(data, output_path):
    """Save point cloud data in HDF5 format with normalization"""
    try:
        import h5py
    except ImportError:
        print("Error: h5py library not found. Please install it with: pip install h5py")
        sys.exit(1)
    
    # Normalize XYZ to unit sphere (-1 to 1)
    normalized_xyz = normalize_to_unit_sphere(data['xyz'])
    
    with h5py.File(output_path, 'w') as f:
        # Save normalized XYZ coordinates
        f.create_dataset('xyz', data=normalized_xyz, compression='gzip')
        
        # Save RGB colors (if available, already normalized to 0-1)
        if data['rgb'] is not None:
            f.create_dataset('rgb', data=data['rgb'], compression='gzip')
        
        # Save intensity (if available, already normalized to 0-1)
        if data['intensity'] is not None:
            f.create_dataset('intensity', data=data['intensity'], compression='gzip')
        
        # Save classification labels (if available)
        if data['labels'] is not None:
            f.create_dataset('labels', data=data['labels'], compression='gzip')
        
        # Add metadata
        f.attrs['n_points'] = len(normalized_xyz)
        f.attrs['has_rgb'] = data['rgb'] is not None
        f.attrs['has_intensity'] = data['intensity'] is not None
        f.attrs['has_labels'] = data['labels'] is not None
        f.attrs['xyz_normalized'] = True
        f.attrs['xyz_range'] = '[-1, 1]'
        f.attrs['rgb_range'] = '[0, 1]' if data['rgb'] is not None else 'N/A'
        f.attrs['intensity_range'] = '[0, 1]' if data['intensity'] is not None else 'N/A'

def convert_las_to_hdf5(input_path, output_path=None):
    """Convert LAS file to normalized HDF5 format"""
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist")
        return False
    
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{input_stem}_normalized.h5")
    
    data = read_las_file(input_path)
    if data is None:
        return False
    
    save_hdf5_format(data, output_path)
    print(f"Saved normalized HDF5 to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert LAS to normalized HDF5 format')
    parser.add_argument('input', help='Input LAS file')
    parser.add_argument('-o', '--output', help='Output HDF5 file')
    
    args = parser.parse_args()
    convert_las_to_hdf5(args.input, args.output)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python las_to_normalized_hdf5.py input.las [-o output.h5]")
    else:
        main()