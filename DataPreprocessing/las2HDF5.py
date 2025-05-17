import os
import laspy
import h5py
import numpy as np

# Path to your folder containing .las files
folder_path = 'Path_To_Folder_With_las_Files'  ## path to folder having .las files
output_hdf5 = 'HFX3D.h5'   #### output hdf5 file name

# Get all .las files
las_files = [f for f in os.listdir(folder_path) if f.endswith('.las')]

# Open HDF5 file
with h5py.File(output_hdf5, 'w') as h5f:
    for las_file in las_files:
        dataset_name = os.path.splitext(las_file)[0]  # Remove the .las extension

        file_path = os.path.join(folder_path, las_file)
        las = laspy.read(file_path)

        # Extract all point dimensions as a structured array
        point_data = las.points.array

        # Convert structured array to a 2D float32 array
        features = np.vstack([point_data[name] for name in point_data.dtype.names]).T
        features = features.astype(np.float32)

        # Store dataset with the same name as the .las file (without extension)
        dset = h5f.create_dataset(dataset_name, data=features)

        # Store feature names as an attribute for reference
        dset.attrs['feature_names'] = ','.join(point_data.dtype.names)

        print(f"Stored {dataset_name} with shape {features.shape} and features: {point_data.dtype.names}")

print(f"\n✅ HDF5 file created: {output_hdf5}")