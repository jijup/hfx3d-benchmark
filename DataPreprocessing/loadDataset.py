import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def preprocess_PCs(hdf5_file, include_gps_time=False, include_intensity=True, normalize_intensity=True):
    data = {}
    feature_names = []

    with h5py.File(hdf5_file, 'r') as h5f:
        for idx, name in enumerate(h5f.keys()):
            points = h5f[name][:]
            raw_feature_names = h5f[name].attrs['feature_names'].split(',')

            # Feature index map
            feat_idx = {feat: i for i, feat in enumerate(raw_feature_names)}

            # Extracting features
            x = points[:, feat_idx['X']]
            y = points[:, feat_idx['Y']]
            z = points[:, feat_idx['Z']]
            classification = points[:, feat_idx['raw_classification']]
            r = points[:, feat_idx['red']]
            g = points[:, feat_idx['green']]
            b = points[:, feat_idx['blue']]

            # Normalize spatial coords
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            z = (z - np.min(z)) / (np.max(z) - np.min(z))

            # Normalize intensity if required
            if include_intensity and 'intensity' in feat_idx:
                intensity = points[:, feat_idx['intensity']]
                if normalize_intensity:
                    intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
            else:
                intensity = None

            # Convert 16-bit RGB to 8-bit
            r = (r / 256)
            g = (g / 256)
            b = (b / 256)

            # Cast classification to int
            classification = classification.astype(np.int32)

            # Create output array with selected features
            features_to_stack = [x, y, z]
            selected_features = ['x', 'y', 'z']

            if include_intensity and intensity is not None:
                features_to_stack.append(intensity)
                selected_features.append('intensity')

            features_to_stack.extend([classification, r, g, b])
            selected_features.extend(['raw_classification', 'red', 'green', 'blue'])

            if include_gps_time and 'gps_time' in feat_idx:
                gps_time = points[:, feat_idx['gps_time']]
                features_to_stack.append(gps_time)
                selected_features.append('gps_time')

            processed = np.stack(features_to_stack, axis=1)
            data[name] = processed

            if idx == 0:
                feature_names = selected_features
                print(f"Selected Features: {feature_names}")

            print(f"{name}: Loaded and processed with shape {processed.shape}")

    return data, feature_names




hdf5_file = "Path_To_Your_HDF5_file"            # Path to HDF5 file
all_buildings, features = preprocess_PCs(hdf5_file, include_gps_time=False, include_intensity=False, normalize_intensity=False)
print("Keys:", all_buildings.keys())


# Print random samples of points with their features
num_samples = 5  # num samples
print("\nPrinting random samples with their features:")
for building_name, building_data in all_buildings.items():
    print(f"\nRandom samples from {building_name}:")
    random_indices = np.random.choice(len(building_data), num_samples, replace=False)
    for idx in random_indices:
        point_str = f"Point {idx}: "
        point_features = [f"{feature.upper()}: {building_data[idx][feat_idx]:.4f}"
                         if feature in ['x', 'y', 'z', 'red', 'green', 'blue']
                         else f"{feature.upper()}: {int(building_data[idx][feat_idx])}"
                         if feature == 'raw_classification'
                         else f"{feature.upper()}: {building_data[idx][feat_idx]}"
                         for feat_idx, feature in enumerate(features)]
        point_str += ", ".join(point_features)
        print(point_str)
    print("-" * 50)