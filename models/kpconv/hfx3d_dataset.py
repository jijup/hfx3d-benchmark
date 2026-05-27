# TODO: Implement HFX3D dataset adapter for KPConv-PyTorch.
#
# This class should inherit from PointCloudDataset (KPConv-PyTorch framework)
# and return the standard 7D feature vector [x, y, z, r, g, b, intensity].
#
# Reference: https://github.com/HuguesTHOMAS/KPConv-PyTorch
#
# Each building is stored as:
#   point_cloud/xyz        (N, 3) float32  -- mean-centered, metric scale
#   point_cloud/rgb        (N, 3) float32  -- [0, 1]
#   point_cloud/intensity  (N,)   float32  -- [0, 1]
#   labels/semantic        (N,)   uint8    -- 0-11
#
# 12 classes: wall, window, door, balcony, vegetation, stairs,
#             terrain, roof, blinds, other, column, arch
#
# See models/randlanet/hfx3d_dataset.py for a reference HDF5 loader.

raise NotImplementedError("hfx3d_dataset.py for KPConv is not yet implemented.")
