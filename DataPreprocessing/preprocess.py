import os
import laspy
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


class LASLabelProcessor:
    def __init__(self, input_folder, output_folder=None, backup=True):
        """
        Initialize the LAS label processor.

        Args:
            input_folder (str): Path to folder containing LAS files
            output_folder (str, optional): Path to output folder. If None, creates 'processed' subfolder
            backup (bool): Whether to create backup of original files
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder) if output_folder else self.input_folder / "processed"
        self.backup = backup

        # Create output directory
        self.output_folder.mkdir(exist_ok=True, parents=True)

        # Create backup directory if needed
        if self.backup:
            self.backup_folder = self.input_folder / "backup"
            self.backup_folder.mkdir(exist_ok=True, parents=True)

    def convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj

    def process_single_file(self, las_path):
        """
        Process a single LAS file: remove label 0 points and shift remaining labels.

        Args:
            las_path (Path): Path to the LAS file

        Returns:
            dict: Statistics about the processing
        """
        try:
            print(f"Processing: {las_path.name}")

            # Read the LAS file
            las_data = laspy.read(las_path)
            original_point_count = len(las_data.points)

            # Get the classification (labels)
            classifications = las_data.classification

            # Count points with label 0
            zero_label_count = np.sum(classifications == 0)

            # Create mask to keep all points except those with label 0
            keep_mask = classifications != 0

            # Filter all point data
            filtered_points = las_data.points[keep_mask]

            # Get the filtered classifications and shift them down by 1
            filtered_classifications = classifications[keep_mask] - 1

            # Create new LAS file with filtered data
            # Start with the header from the original file
            new_header = laspy.LasHeader(
                point_format=las_data.header.point_format,
                version=las_data.header.version
            )

            # Copy header information
            new_header.scales = las_data.header.scales
            new_header.offsets = las_data.header.offsets

            # Create new LAS data object
            new_las = laspy.LasData(new_header)

            # Copy all point data
            new_las.points = filtered_points

            # Update the classification with shifted labels
            new_las.classification = filtered_classifications

            # Update point count in header
            new_las.header.point_count = len(filtered_points)

            # Generate output path
            output_path = self.output_folder / las_path.name

            # Create backup if requested
            if self.backup:
                backup_path = self.backup_folder / las_path.name
                if not backup_path.exists():
                    shutil.copy2(las_path, backup_path)
                    print(f"  Backup created: {backup_path}")

            # Write the processed file
            new_las.write(output_path)

            # Calculate statistics
            final_point_count = len(filtered_points)
            unique_labels_original = np.unique(classifications)
            unique_labels_new = np.unique(filtered_classifications)

            stats = {
                'file': las_path.name,
                'original_points': int(original_point_count),  # Convert to Python int
                'removed_points': int(zero_label_count),  # Convert to Python int
                'final_points': int(final_point_count),  # Convert to Python int
                'original_labels': [int(x) for x in unique_labels_original],  # Convert numpy array to list of ints
                'new_labels': [int(x) for x in unique_labels_new],  # Convert numpy array to list of ints
                'success': True
            }

            print(f"  Original points: {original_point_count:,}")
            print(f"  Removed points (label 0): {zero_label_count:,}")
            print(f"  Final points: {final_point_count:,}")
            print(f"  Original labels: {unique_labels_original}")
            print(f"  New labels: {unique_labels_new}")
            print(f"  Saved to: {output_path}")

            return stats

        except Exception as e:
            print(f"Error processing {las_path.name}: {str(e)}")
            return {
                'file': las_path.name,
                'error': str(e),
                'success': False
            }

    def process_all_files(self):
        """
        Process all LAS files in the input folder.

        Returns:
            list: List of processing statistics for each file
        """
        # Find all LAS files
        las_files = list(self.input_folder.glob("*.las"))

        if not las_files:
            print(f"No LAS files found in {self.input_folder}")
            return []

        print(f"Found {len(las_files)} LAS files to process")
        print(f"Output folder: {self.output_folder}")
        if self.backup:
            print(f"Backup folder: {self.backup_folder}")
        print("-" * 50)

        # Process each file
        all_stats = []
        successful_files = 0

        for las_path in tqdm(las_files, desc="Processing files"):
            stats = self.process_single_file(las_path)
            all_stats.append(stats)

            if stats['success']:
                successful_files += 1

            print()  # Add spacing between files

        # Print summary
        print("=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {len(las_files)}")
        print(f"Successful: {successful_files}")
        print(f"Failed: {len(las_files) - successful_files}")

        # Calculate overall statistics
        total_original_points = sum(s.get('original_points', 0) for s in all_stats if s['success'])
        total_removed_points = sum(s.get('removed_points', 0) for s in all_stats if s['success'])
        total_final_points = sum(s.get('final_points', 0) for s in all_stats if s['success'])

        print(f"\nOverall point statistics:")
        print(f"  Total original points: {total_original_points:,}")
        print(f"  Total removed points: {total_removed_points:,}")
        print(f"  Total final points: {total_final_points:,}")
        print(f"  Removal percentage: {(total_removed_points / total_original_points) * 100:.2f}%")

        # Show label transformation summary
        print(f"\nLabel transformation:")
        print(f"  Before: All labels (including 0)")
        print(f"  After: Labels shifted down by 1 (0 removed, others decremented)")

        return all_stats

    def validate_processing(self):
        """
        Validate that the processing was done correctly by comparing a sample file.
        """
        las_files = list(self.input_folder.glob("*.las"))
        if not las_files:
            print("No files to validate")
            return

        # Take the first file as sample
        sample_file = las_files[0]
        processed_file = self.output_folder / sample_file.name

        if not processed_file.exists():
            print(f"Processed file {processed_file} not found for validation")
            return

        print(f"\nValidating processing with sample file: {sample_file.name}")
        print("-" * 50)

        # Read both files
        original = laspy.read(sample_file)
        processed = laspy.read(processed_file)

        # Check classifications
        orig_labels = np.unique(original.classification)
        proc_labels = np.unique(processed.classification)

        print(f"Original labels: {orig_labels}")
        print(f"Processed labels: {proc_labels}")

        # Verify that processed labels are original labels - 1 (excluding 0)
        expected_labels = orig_labels[orig_labels != 0] - 1

        if np.array_equal(np.sort(proc_labels), np.sort(expected_labels)):
            print("✓ Label transformation verified correctly")
        else:
            print("✗ Label transformation may have issues")
            print(f"Expected: {np.sort(expected_labels)}")
            print(f"Got: {np.sort(proc_labels)}")

        # Check point count
        orig_non_zero = np.sum(original.classification != 0)
        if len(processed.points) == orig_non_zero:
            print("✓ Point count verified correctly")
        else:
            print("✗ Point count mismatch")
            print(f"Expected: {orig_non_zero}")
            print(f"Got: {len(processed.points)}")


def main():
    # Configuration
    input_folder = r"C:\Users\s4824030\PycharmProjects\hfx3d-benchmark\data\WACV"
    output_folder = None  # Will create 'processed' subfolder in input_folder

    # Create processor
    processor = LASLabelProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
        backup=True  # Set to False if you don't want backups
    )

    # Process all files
    stats = processor.process_all_files()

    # Validate the processing
    processor.validate_processing()

    # Optionally save statistics to file
    import json
    stats_file = processor.output_folder / "processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nDetailed statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()