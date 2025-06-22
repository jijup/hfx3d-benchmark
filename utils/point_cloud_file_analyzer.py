import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 120)
pd.set_option('display.max_rows', 20)

def read_xyz_file(file_path):
    """Read XYZ file and return DataFrame with dynamic column detection."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        data_start_line = 0
        header_line = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('//'):
                if any(keyword in line.upper() for keyword in ['X', 'Y', 'Z', 'R', 'G', 'B', 'RED', 'GREEN', 'BLUE']):
                    header_line = line[2:].strip()
                continue
            elif line and not line.startswith('#'):
                data_start_line = i
                break
        
        if header_line:
            column_names = header_line.split()
            df = pd.read_csv(file_path, sep=r'\s+', header=None, 
                           skiprows=data_start_line, low_memory=False)
            
            if len(column_names) == df.shape[1]:
                df.columns = column_names
            else:
                df.columns = [f'column_{i}' for i in range(df.shape[1])]
                print(f"Warning: Header has {len(column_names)} columns but data has {df.shape[1]} columns. Using generic names.")
        else:
            df = pd.read_csv(file_path, sep=r'\s+', header=None, 
                           skiprows=data_start_line, low_memory=False)
            
            first_row_numeric = True
            try:
                pd.to_numeric(df.iloc[0])
            except (ValueError, TypeError):
                first_row_numeric = False
            
            if not first_row_numeric:
                df = pd.read_csv(file_path, sep=r'\s+', skiprows=data_start_line, low_memory=False)
            else:
                df.columns = [f'column_{i}' for i in range(df.shape[1])]
        
        return df, "XYZ"
    except Exception as e:
        print(f"Error reading XYZ file: {e}")
        return None, None

def read_pcd_file(file_path):
    """Read PCD file and return DataFrame with original field names."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        header_end = 0
        fields = []
        data_format = "ascii"
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('DATA'):
                data_format = line.split()[1].lower()
                header_end = i + 1
                break
        
        if data_format != "ascii":
            raise ValueError("Only ASCII format PCD files are supported")
        
        data_lines = lines[header_end:]
        data = []
        for line in data_lines:
            line = line.strip()
            if line:
                values = line.split()
                data.append([float(v) for v in values])
        
        df = pd.DataFrame(data, columns=fields)
        return df, "PCD"
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        return None, None

def read_ply_file(file_path):
    """Read PLY file and return DataFrame with properties from header."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        header_end = 0
        properties = []
        num_vertices = 0
        data_format = "ascii"
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('format'):
                format_parts = line.split()
                if len(format_parts) >= 2:
                    data_format = format_parts[1].lower()
            elif line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_name, prop_type))
            elif line == 'end_header':
                header_end = i + 1
                break
        
        if data_format not in ["ascii", "ascii_1.0"]:
            raise ValueError("Only ASCII format PLY files are supported")
        
        if num_vertices == 0:
            raise ValueError("No vertex elements found in PLY header")
        
        # Read vertex data
        data_lines = lines[header_end:header_end + num_vertices]
        data = []
        
        for line in data_lines:
            line = line.strip()
            if line:
                values = line.split()
                parsed_values = []
                for j, (prop_name, prop_type) in enumerate(properties):
                    if j < len(values):
                        try:
                            if prop_type in ['uchar', 'char', 'int', 'uint']:
                                parsed_values.append(int(values[j]))
                            else:  # float, double
                                parsed_values.append(float(values[j]))
                        except (ValueError, IndexError):
                            parsed_values.append(0)
                    else:
                        parsed_values.append(0)
                data.append(parsed_values)
        
        # Create DataFrame with property names
        column_names = [prop[0] for prop in properties]
        df = pd.DataFrame(data, columns=column_names)
        
        # Add PLY metadata
        df._ply_format = data_format
        df._ply_num_vertices = num_vertices
        df._ply_properties = properties
        
        return df, "PLY"
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return None, None

def read_las_file(file_path):
    """Read LAS file and return DataFrame with all available dimensions."""
    try:
        import laspy
        las = laspy.read(file_path)
        
        num_points = len(las.points)
        num_sample_points = min(100000, num_points)
        
        if num_points > num_sample_points:
            sample_indices = np.random.choice(num_points, num_sample_points, replace=False)
        else:
            sample_indices = np.arange(num_points)
        
        points_data = {}
        for dim in las.point_format.dimension_names:
            if dim.lower() in ['x', 'y', 'z']:
                points_data[dim] = getattr(las, dim.lower())[sample_indices]
            else:
                points_data[dim] = las.points[dim][sample_indices]
        
        df = pd.DataFrame(points_data)
        df._las_total_points = num_points
        df._las_dimensions = list(las.point_format.dimension_names)
        df._las_is_sampled = num_points > num_sample_points
        
        return df, "LAS"
    except ImportError:
        print("Error: laspy library not found. Please install it with: pip install laspy")
        return None, None
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        return None, None

def read_hdf5_file(file_path):
    """Read HDF5 file and return DataFrame with feature names from attributes."""
    try:
        import h5py
        
        datasets_info = {}
        total_points = 0
        
        with h5py.File(file_path, 'r') as h5f:
            dataset_names = list(h5f.keys())
            
            if not dataset_names:
                raise ValueError("No datasets found in HDF5 file")
            
            print(f"Found {len(dataset_names)} datasets: {dataset_names}")
            
            # Combine all datasets into a single DataFrame for analysis
            all_data = []
            feature_names = None
            
            for dataset_name in dataset_names:
                dataset = h5f[dataset_name]
                points = dataset[:]
                
                # Get feature names from attributes
                if 'feature_names' in dataset.attrs:
                    current_features = dataset.attrs['feature_names']
                    if isinstance(current_features, bytes):
                        current_features = current_features.decode('utf-8')
                    current_features = current_features.split(',')
                    
                    if feature_names is None:
                        feature_names = current_features
                    elif feature_names != current_features:
                        print(f"Warning: Dataset {dataset_name} has different features: {current_features}")
                
                datasets_info[dataset_name] = {
                    'shape': points.shape,
                    'features': current_features if 'current_features' in locals() else None
                }
                
                # Sample large datasets to avoid memory issues
                if len(points) > 50000:
                    sample_indices = np.random.choice(len(points), 50000, replace=False)
                    points = points[sample_indices]
                
                all_data.append(points)
                total_points += len(dataset)
            
            # Combine all sampled data
            combined_data = np.vstack(all_data)
            
            # Create DataFrame
            if feature_names:
                df = pd.DataFrame(combined_data, columns=feature_names)
            else:
                df = pd.DataFrame(combined_data, columns=[f'column_{i}' for i in range(combined_data.shape[1])])
                print("Warning: No feature names found in attributes, using generic column names")
            
            # Add metadata
            df._hdf5_datasets = datasets_info
            df._hdf5_total_points = total_points
            df._hdf5_is_sampled = any(info['shape'][0] > 50000 for info in datasets_info.values())
            
            return df, "HDF5"
            
    except ImportError:
        print("Error: h5py library not found. Please install it with: pip install h5py")
        return None, None
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return None, None

def detect_column_types(df):
    """Detect column types based on data patterns and common naming conventions."""
    column_info = {}
    
    for col in df.columns:
        col_lower = col.lower()
        data_type = 'other'
        
        if col_lower in ['x', 'y', 'z'] or (col.startswith('column_') and col in ['column_0', 'column_1', 'column_2']):
            data_type = 'coordinate'
        elif col_lower in ['red', 'green', 'blue', 'r', 'g', 'b'] or (col.startswith('column_') and col in ['column_3', 'column_4', 'column_5'] and len(df.columns) >= 6):
            data_type = 'color'
        elif 'intensity' in col_lower or (col == 'column_6' and len(df.columns) >= 7):
            data_type = 'intensity'
        elif any(keyword in col_lower for keyword in ['classification', 'class', 'label', 'raw_classification']) or (col == 'column_7' and len(df.columns) >= 8):
            data_type = 'classification'
        elif any(keyword in col_lower for keyword in ['normal_x', 'normal_y', 'normal_z', 'nx', 'ny', 'nz']):
            data_type = 'normal'
        elif col_lower in ['return_number', 'number_of_returns', 'scan_angle', 'scan_direction_flag', 'edge_of_flight_line', 'gps_time']:
            data_type = 'lidar_attribute'
        
        if data_type == 'other' and col.startswith('//'):
            clean_col = col.replace('//', '').lower()
            if clean_col in ['x', 'y', 'z']:
                data_type = 'coordinate'
            elif clean_col in ['red', 'green', 'blue', 'r', 'g', 'b']:
                data_type = 'color'
            elif 'intensity' in clean_col:
                data_type = 'intensity'
            elif 'classification' in clean_col or 'class' in clean_col:
                data_type = 'classification'
        
        column_info[col] = {
            'type': data_type,
            'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
            'unique_count': df[col].nunique(),
            'has_nulls': df[col].isnull().any()
        }
    
    return column_info

def get_columns_by_type(column_info, data_type):
    """Get columns of a specific type."""
    return [col for col, info in column_info.items() if info['type'] == data_type]

def analyze_columns_intelligently(df, column_info):
    """Analyze columns based on detected types and return organized information."""
    analysis = {
        'coordinates': get_columns_by_type(column_info, 'coordinate'),
        'colors': get_columns_by_type(column_info, 'color'),
        'intensity': get_columns_by_type(column_info, 'intensity'),
        'classification': get_columns_by_type(column_info, 'classification'),
        'normals': get_columns_by_type(column_info, 'normal'),
        'lidar_attributes': get_columns_by_type(column_info, 'lidar_attribute'),
        'other': get_columns_by_type(column_info, 'other')
    }
    
    priority_columns = []
    for col_type in ['coordinates', 'colors', 'intensity', 'classification', 'normals', 'lidar_attributes']:
        priority_columns.extend(analysis[col_type])
    
    remaining_space = 10 - len(priority_columns)
    if remaining_space > 0:
        priority_columns.extend(analysis['other'][:remaining_space])
    
    return analysis, priority_columns

def get_friendly_column_name(col, col_type, position_in_type):
    """Generate user-friendly names for display purposes."""
    display_col = col.replace('//', '') if col.startswith('//') else col
    
    if col.startswith('column_') or col.startswith('//'):
        if col_type == 'coordinate':
            coord_names = ['X', 'Y', 'Z']
            if position_in_type < len(coord_names):
                return f"{coord_names[position_in_type]} ({display_col})"
        elif col_type == 'color':
            color_names = ['Red', 'Green', 'Blue']
            if position_in_type < len(color_names):
                return f"{color_names[position_in_type]} ({display_col})"
        elif col_type == 'intensity':
            return f"Intensity ({display_col})"
        elif col_type == 'classification':
            return f"Classification ({display_col})"
        return f"{display_col} ({col_type})"
    return display_col

def display_columns_with_friendly_names(df, analysis, column_info):
    """Display column information with user-friendly names."""
    print("\n=== Available Dimensions by Type ===")
    for col_type, columns in analysis.items():
        if columns:
            print(f"\n{col_type.replace('_', ' ').title()}:")
            for i, col in enumerate(columns):
                friendly_name = get_friendly_column_name(col, col_type, i)
                info = column_info[col]
                type_info = f"({info['unique_count']} unique values)" if info['type'] == 'classification' else ""
                null_info = " [has nulls]" if info['has_nulls'] else ""
                print(f"  {i+1:2d}. {friendly_name} {type_info}{null_info}")

def display_hdf5_info(df):
    """Display HDF5-specific information."""
    if hasattr(df, '_hdf5_datasets'):
        print(f"\n=== HDF5 Dataset Information ===")
        for dataset_name, info in df._hdf5_datasets.items():
            print(f"Dataset '{dataset_name}': {info['shape'][0]:,} points, {info['shape'][1]} features")
            if info['features']:
                print(f"  Features: {', '.join(info['features'])}")

def display_ply_info(df):
    """Display PLY-specific information."""
    if hasattr(df, '_ply_properties'):
        print(f"\n=== PLY File Information ===")
        print(f"Format: {df._ply_format}")
        print(f"Vertices: {df._ply_num_vertices:,}")
        print("Properties:")
        for prop_name, prop_type in df._ply_properties:
            print(f"  {prop_name}: {prop_type}")

def process_point_cloud(file_path):
    """Process point cloud file and print statistics."""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.xyz':
        df, file_type = read_xyz_file(file_path)
    elif file_ext == '.pcd':
        df, file_type = read_pcd_file(file_path)
    elif file_ext == '.ply':
        df, file_type = read_ply_file(file_path)
    elif file_ext == '.las':
        df, file_type = read_las_file(file_path)
    elif file_ext in ['.h5', '.hdf5']:
        df, file_type = read_hdf5_file(file_path)
    else:
        print(f"Error: Unsupported file format '{file_ext}'. Supported formats: .xyz, .pcd, .ply, .las, .h5, .hdf5")
        return
    
    if df is None:
        return
    
    column_info = detect_column_types(df)
    analysis, priority_columns = analyze_columns_intelligently(df, column_info)
    
    if file_type == "LAS":
        total_points = df._las_total_points
        is_sampled = df._las_is_sampled
        print(f"Successfully loaded LAS file with {total_points:,} points")
        if is_sampled:
            print(f"Using sample of {len(df):,} points for analysis")
    elif file_type == "HDF5":
        total_points = df._hdf5_total_points
        is_sampled = df._hdf5_is_sampled
        print(f"Successfully loaded HDF5 file with {total_points:,} total points across all datasets")
        if is_sampled:
            print(f"Using combined sample of {len(df):,} points for analysis")
        display_hdf5_info(df)
    elif file_type == "PLY":
        total_points = len(df)
        is_sampled = False
        print(f"Successfully loaded PLY file with {total_points:,} points")
        display_ply_info(df)
    else:
        total_points = len(df)
        is_sampled = False
        print(f"Successfully loaded {file_type} file with {len(df)} points")
    
    display_columns_with_friendly_names(df, analysis, column_info)
    
    num_sample_points = min(10000, len(df))
    if len(df) > num_sample_points:
        sample_df = df.sample(n=num_sample_points, random_state=42)
        display_sample_info = True
    else:
        sample_df = df
        display_sample_info = False
    
    display_points = sample_df.sample(n=min(10, len(sample_df)), random_state=42)
    sample_desc = "sample" if (file_type == "LAS" and is_sampled) or (file_type == "HDF5" and is_sampled) or display_sample_info else "dataset"
    
    print(f"\n=== Sample Points (10 random from {sample_desc}) ===")
    
    display_columns = [col for col in priority_columns if col in df.columns][:10]
    
    if display_columns:
        print(display_points[display_columns].to_string(
            index=True,
            float_format=lambda x: '{:.3f}'.format(x) if isinstance(x, float) else str(x),
            col_space=12
        ))
    else:
        print("No suitable columns found for display")
    
    numeric_cols = [col for col in df.columns if column_info[col]['is_numeric']]
    if numeric_cols:
        range_stats = pd.DataFrame({
            'Minimum': sample_df[numeric_cols].min(),
            'Maximum': sample_df[numeric_cols].max(),
            'Range': sample_df[numeric_cols].max() - sample_df[numeric_cols].min()
        })
        
        sample_info = " (from sample)" if (file_type == "LAS" and is_sampled) or (file_type == "HDF5" and is_sampled) or display_sample_info else ""
        print(f"\n=== Numeric Channel Statistics{sample_info} ===")
        print(range_stats.to_string(float_format=lambda x: '{:,.3f}'.format(x)))
    
    classification_cols = analysis['classification']
    if classification_cols:
        print(f"\n=== Classification Distribution ===")
        for col in classification_cols:
            if col in sample_df.columns:
                friendly_name = get_friendly_column_name(col, 'classification', 0)
                class_counts = sample_df[col].value_counts().sort_index()
                print(f"\n{friendly_name}:")
                for class_val, count in class_counts.items():
                    percentage = (count / len(sample_df)) * 100
                    print(f"  Class {class_val}: {count:,} points ({percentage:.2f}%)")
    
    color_cols = analysis['colors']
    if color_cols:
        print(f"\n=== Color Channel Statistics ===")
        for i, col in enumerate(color_cols):
            if col in sample_df.columns:
                friendly_name = get_friendly_column_name(col, 'color', i)
                color_stats = sample_df[col].describe()
                print(f"\n{friendly_name}:")
                print(f"  Min: {color_stats['min']:.3f}, Max: {color_stats['max']:.3f}")
                print(f"  Mean: {color_stats['mean']:.3f}, Std: {color_stats['std']:.3f}")
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total number of points in file: {total_points:,}")
    if (file_type == "LAS" and is_sampled) or (file_type == "HDF5" and is_sampled) or display_sample_info:
        print(f"Number of points in analysis sample: {len(sample_df):,}")
    print(f"Number of channels: {len(df.columns)}")
    print(f"File type: {file_type}")
    
    print(f"\nColumn breakdown:")
    for col_type, columns in analysis.items():
        if columns:
            print(f"  {col_type.replace('_', ' ').title()}: {len(columns)} columns")

def main():
    parser = argparse.ArgumentParser(description='Analyze XYZ, PCD, PLY, LAS, and HDF5 point cloud files with dynamic column detection')
    parser.add_argument('files', nargs='+', help='Path(s) to XYZ, PCD, PLY, LAS, or HDF5 file(s)')
    
    args = parser.parse_args()
    
    for file_path in args.files:
        process_point_cloud(file_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Point Cloud File Analyzer")
        print("Dynamic column detection for XYZ, RGB, intensity, classification and other data")
        print("Supported formats: .xyz, .pcd, .ply, .las, .h5, .hdf5")
        print("\nEnter file path(s) (space-separated for multiple files):")
        file_paths = input().strip().split()
        
        for file_path in file_paths:
            process_point_cloud(file_path)
    else:
        main()