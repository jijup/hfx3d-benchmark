import os
import re
import laspy
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
from pathlib import Path

class TrainTestSplitAnalyzer:
    def __init__(self, data_folder, output_dir=None):
        self.data_folder = Path(data_folder)
        self.output_dir = Path(output_dir) if output_dir else Path("split_analysis_output")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Define the train/test split based on building numbers
        self.train_buildings = ["001", "004", "005", "006", "007", "009", "012"]
        self.test_buildings = ["002", "003", "008", "010", "011"]

        # Load label map
        self.label_map = self.load_label_map()
        self.id_to_name = {v: k for k, v in self.label_map.items()}

        # Set plotting style
        plt.style.use('seaborn-v0_8-pastel')
        sns.set_context("notebook", font_scale=1.2)
        self.colors = {
            'train': '#3498db',  # Blue
            'test': '#e74c3c'    # Red
        }

    def format_number(self, num):
        """Format numbers for display with appropriate suffixes."""
        if num == 0:
            return "0"
        elif num < 1000:
            return f"{int(num)}"
        elif num < 1000000:
            return f"{num/1000:.1f}K"
        elif num < 1000000000:
            return f"{num/1000000:.1f}M"
        else:
            return f"{num/1000000000:.1f}B"

    def load_label_map(self, json_path=r"C:\Users\s4824030\PycharmProjects\hfx3d-benchmark\configs\label_map.json"):
        """Load the label map from JSON file."""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "wall": 0, "window": 1, "door": 2,
                "balcony": 3, "vegetation": 4, "stairs": 5, "terrain": 6,
                "roof": 7, "blinds": 8, "other": 9, "column": 10, "arch": 11
            }

    def is_train_file(self, filename):
        """Check if the filename belongs to train set based on building number."""
        for building_num in self.train_buildings:
            if building_num in filename:
                return True
        return False

    def is_test_file(self, filename):
        """Check if the filename belongs to test set based on building number."""
        for building_num in self.test_buildings:
            if building_num in filename:
                return True
        return False

    def analyze_dataset(self):
        """Analyze the dataset and collect statistics about class distribution."""
        las_files = list(self.data_folder.glob("*.las"))

        if not las_files:
            print(f"No .las files found in {self.data_folder}")
            return None

        # Initialize data structures
        train_files = []
        test_files = []
        train_class_counts = defaultdict(int)
        test_class_counts = defaultdict(int)
        per_file_counts = {}

        # Process each file
        for las_path in las_files:
            filename = las_path.name

            # Skip files that don't match our train/test criteria
            if not (self.is_train_file(filename) or self.is_test_file(filename)):
                print(f"Skipping {filename} - doesn't match train/test criteria")
                continue

            print(f"Processing: {filename}")

            try:
                # Read LAS file
                las_data = laspy.read(las_path)
                class_counts = dict(Counter(las_data.classification))
                per_file_counts[filename] = class_counts

                # Add to appropriate split
                if self.is_train_file(filename):
                    train_files.append(filename)
                    for class_id, count in class_counts.items():
                        train_class_counts[class_id] += count
                else:
                    test_files.append(filename)
                    for class_id, count in class_counts.items():
                        test_class_counts[class_id] += count

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        # Combine results
        return {
            'train_files': train_files,
            'test_files': test_files,
            'train_class_counts': dict(train_class_counts),
            'test_class_counts': dict(test_class_counts),
            'per_file_counts': per_file_counts
        }

    def visualize_split(self, analysis_results):
        """Create visualizations comparing train/test class distributions."""
        if not analysis_results:
            print("No analysis results to visualize")
            return

        # Only keep the Train_Test_Splits_HFX3D visualization
        self._plot_class_distribution_comparison(analysis_results)

        # Generate detailed report
        self._generate_split_report(analysis_results)

    def _plot_class_distribution_comparison(self, results):
        """Create horizontal bar chart comparing train/test distributions."""
        train_counts = results['train_class_counts']
        test_counts = results['test_class_counts']

        # Get all class IDs
        all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))

        # Define color palette for train/test splits
        palette_dict = {'Train': '#1f77b4', 'Test': '#d62728'}

        # Create DataFrame for easier plotting
        data = []
        for class_id in all_classes:
            class_name = self.id_to_name.get(class_id, f'Unknown-{class_id}')
            train_count = train_counts.get(class_id, 0)
            test_count = test_counts.get(class_id, 0)

            data.append({
                'Class': class_name.capitalize(),
                'Split': 'Train',
                'Count': train_count,
                'LogCount': np.log10(train_count + 1)  # +1 to handle zeros
            })

            data.append({
                'Class': class_name.capitalize(),
                'Split': 'Test',
                'Count': test_count,
                'LogCount': np.log10(test_count + 1)
            })

        df = pd.DataFrame(data)

        # Create figure with larger size and adjusted margins
        plt.figure(figsize=(16, 12))

        # Sort classes by total count (descending)
        class_order = df.groupby('Class')['Count'].sum().sort_values(ascending=False).index

        # Create horizontal bar chart with log scale
        ax = plt.gca()
        sns.barplot(
            x='Count', 
            y='Class', 
            hue='Split',
            data=df,
            palette=palette_dict,
            order=class_order,
            orient='h'
        )

        # Set logarithmic scale for x-axis starting from 10^5
        ax.set_xscale('log')
        ax.set_xlim(100000, None)  # Start from 10^5 (100,000)

        # Add count labels on bars
        for i, p in enumerate(plt.gca().patches):
            width = p.get_width()
            if width >= 100000:  # Only show labels for values within our scale range
                plt.text(
                    width * 1.1,  # Position slightly to the right of the bar
                    p.get_y() + p.get_height()/2,
                    self.format_number(width),
                    ha='left', 
                    va='center',
                    fontsize=10
                )

        # Format x-axis labels
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: self.format_number(x)))

        plt.title('Train vs Test Splits HFX3D', fontsize=16, pad=20)
        plt.xlabel('Number of Points (log scale)', fontsize=12)
        plt.ylabel('Class', fontsize=12)

        # Use subplots_adjust instead of tight_layout for better control
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

        plt.savefig(self.output_dir / 'Train_Test_Splits_HFX3D.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_normalized_distribution(self, results):
        """Create horizontal stacked bar chart showing percentage distribution."""
        train_counts = results['train_class_counts']
        test_counts = results['test_class_counts']

        # Calculate total points in each split
        train_total = sum(train_counts.values())
        test_total = sum(test_counts.values())

        # Get all class IDs
        all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))

        # Create percentage data
        train_pct = {class_id: (count / train_total) * 100 for class_id, count in train_counts.items()}
        test_pct = {class_id: (count / test_total) * 100 for class_id, count in test_counts.items()}

        # Create DataFrame
        data = []
        for class_id in all_classes:
            class_name = self.id_to_name.get(class_id, f'Unknown-{class_id}')
            train_percentage = train_pct.get(class_id, 0)
            test_percentage = test_pct.get(class_id, 0)

            data.append({
                'Class': class_name.capitalize(),
                'Split': 'Train',
                'Percentage': train_percentage
            })

            data.append({
                'Class': class_name.capitalize(),
                'Split': 'Test',
                'Percentage': test_percentage
            })

        df = pd.DataFrame(data)

        # Create figure
        plt.figure(figsize=(14, 10))

        # Sort classes by average percentage (descending)
        class_order = df.groupby('Class')['Percentage'].mean().sort_values(ascending=False).index

        # Create horizontal bar chart
        # Fix: Pass palette as dictionary correctly
        palette_dict = {'Train': self.colors['train'], 'Test': self.colors['test']}

        sns.barplot(
            x='Percentage', 
            y='Class', 
            hue='Split',
            data=df,
            palette=palette_dict,
            order=class_order,
            orient='h'
        )

        # Modify the text placement
        for i, p in enumerate(plt.gca().patches):
            width = p.get_width()
            if width > 0.5:  # Only add label if percentage is significant
                plt.text(
                    width/2,  # Position text in middle of bar
                    p.get_y() + p.get_height()/2,
                    f'{width:.1f}%',
                    ha='center',  # Center align the text
                    va='center'
                )

        plt.title('Train vs Test Class Distribution (Percentage)', fontsize=16, pad=20)
        plt.xlabel('Percentage of Points', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution_percentage.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_split_report(self, results):
        """Generate a detailed text report about the train/test split."""
        train_counts = results['train_class_counts']
        test_counts = results['test_class_counts']

        # Calculate totals
        train_total = sum(train_counts.values())
        test_total = sum(test_counts.values())
        overall_total = train_total + test_total

        # Get all classes
        all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))

        check_mark = "[OK]"
        warning_mark = "[WARN]"
        error_mark = "[ERROR]"
        bullet_point = "-"

        with open(self.output_dir / 'split_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("Train/Test Split Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            # File distribution
            f.write("File Distribution:\n")
            f.write(f"  Train files ({len(results['train_files'])}): {', '.join(results['train_files'])}\n")
            f.write(f"  Test files ({len(results['test_files'])}): {', '.join(results['test_files'])}\n\n")

            # Point distribution
            f.write("Point Distribution:\n")
            f.write(f"  Train points: {train_total:,} ({train_total/overall_total:.1%} of total)\n")
            f.write(f"  Test points: {test_total:,} ({test_total/overall_total:.1%} of total)\n")
            f.write(f"  Total points: {overall_total:,}\n\n")

            # Class distribution table
            f.write("Class Distribution:\n")
            f.write(f"{'Class':<15} {'Train Count':>12} {'Train %':>8} {'Test Count':>12} {'Test %':>8} {'Total':>12} {'Train/Test':>10}\n")
            f.write("-" * 75 + "\n")

            for class_id in all_classes:
                class_name = self.id_to_name.get(class_id, f'Unknown-{class_id}')
                train_count = train_counts.get(class_id, 0)
                test_count = test_counts.get(class_id, 0)
                total_count = train_count + test_count

                train_pct = (train_count / train_total) * 100 if train_total else 0
                test_pct = (test_count / test_total) * 100 if test_total else 0

                # Calculate train/test ratio
                if test_count == 0:
                    ratio = "inf"
                elif train_count == 0:
                    ratio = "0"
                else:
                    ratio = f"{train_count / test_count:.2f}"

                f.write(f"{class_name:<15} {train_count:>12,} {train_pct:>7.1f}% {test_count:>12,} {test_pct:>7.1f}% {total_count:>12,} {ratio:>10}\n")

            # Split analysis
            f.write("\nSplit Quality Analysis:\n")
            f.write("=" * 50 + "\n")

            # Check for missing classes
            missing_in_train = [self.id_to_name.get(c, f'Unknown-{c}') for c in all_classes if c not in train_counts]
            missing_in_test = [self.id_to_name.get(c, f'Unknown-{c}') for c in all_classes if c not in test_counts]

            if missing_in_train:
                f.write(f"{warning_mark} WARNING: Classes missing from train set: {', '.join(missing_in_train)}\n")
            else:
                f.write(f"{check_mark} All classes represented in train set\n")

            if missing_in_test:
                f.write(f"{warning_mark} WARNING: Classes missing from test set: {', '.join(missing_in_test)}\n")
            else:
                f.write(f"{check_mark} All classes represented in test set\n")

            # Check for class distribution similarity
            f.write("\nClass Distribution Similarity:\n")
            similarity_score = 0

            for class_id in all_classes:
                train_pct = (train_counts.get(class_id, 0) / train_total) * 100 if train_total else 0
                test_pct = (test_counts.get(class_id, 0) / test_total) * 100 if test_total else 0
                difference = abs(train_pct - test_pct)
                similarity_score += difference

                # Highlight significant differences
                if difference > 5:
                    class_name = self.id_to_name.get(class_id, f'Unknown-{class_id}')
                    f.write(f"  {bullet_point} {class_name}: {train_pct:.1f}% in train vs {test_pct:.1f}% in test (diff: {difference:.1f}%)\n")

            # Overall similarity score (lower is better)
            normalized_score = similarity_score / len(all_classes)
            f.write(f"\nOverall distribution difference: {similarity_score:.1f}% ({normalized_score:.1f}% per class)\n")

            if normalized_score < 2:
                f.write(f"{check_mark} Excellent distribution similarity between train and test\n")
            elif normalized_score < 5:
                f.write(f"{check_mark} Good distribution similarity between train and test\n")
            elif normalized_score < 10:
                f.write(f"{warning_mark} Fair distribution similarity - consider rebalancing\n")
            else:
                f.write(f"{error_mark} Poor distribution similarity - strongly consider rebalancing\n")

        print(f"Analysis complete! Results saved to {self.output_dir}")

    def _plot_class_train_test_split_ratio(self, results):
        """Create horizontal bar chart showing how each class is split between train/test."""
        train_counts = results['train_class_counts']
        test_counts = results['test_class_counts']

        # Get all class IDs
        all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))

        # Define color palette for train/test splits
        palette_dict = {'Train': '#2E86AB', 'Test': '#A23B72'}

        # Create DataFrame with percentage of each class in train vs test
        data = []
        for class_id in all_classes:
            class_name = self.id_to_name.get(class_id, f'Unknown-{class_id}')
            train_count = train_counts.get(class_id, 0)
            test_count = test_counts.get(class_id, 0)
            total_count = train_count + test_count

            # Calculate what percentage of this class is in train vs test
            train_percentage = (train_count / total_count) * 100 if total_count else 0
            test_percentage = (test_count / total_count) * 100 if total_count else 0

            data.append({
                'Class': class_name.capitalize(),
                'Split': 'Train',
                'Percentage': train_percentage,
                'Count': train_count,
                'Total': total_count
            })

            data.append({
                'Class': class_name.capitalize(),
                'Split': 'Test',
                'Percentage': test_percentage,
                'Count': test_count, 
                'Total': total_count
            })

        df = pd.DataFrame(data)

        # Sort classes by total count (descending)
        class_order = df.groupby('Class')['Count'].sum().sort_values(ascending=False).index

        # Create figure with two x-axes
        fig, ax1 = plt.subplots(figsize=(14, 10))

        # Create the percentage plot
        sns.barplot(
            x='Percentage', 
            y='Class', 
            hue='Split',
            data=df,
            palette=palette_dict,
            order=class_order,
            orient='h',
            ax=ax1
        )

        # Add a second axes for counts (on the right side)
        ax2 = ax1.twiny()  # Create a second x-axis sharing the same y-axis

        # Get total counts for each class
        total_counts = df.groupby('Class')['Total'].max().reindex(class_order)

        # Plot the counts on the top axis with log scale
        ax2.set_xscale('log')
        ax2.set_xlim(100, max(total_counts) * 1.5)

        # Format the top axis labels
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax2.set_xlabel('Total Points (log scale)', fontsize=12)

        # Format the bottom axis
        ax1.set_xlim(0, 105)
        ax1.set_xlabel('Percentage of Class in Each Split', fontsize=12)

        # Add percentage labels on bars
        for i, p in enumerate(ax1.patches):
            width = p.get_width()
            if width > 2:  # Only add label if percentage is significant
                plt.text(
                    width/2,
                    p.get_y() + p.get_height()/2,
                    f'{width:.1f}%',
                    ha='center',
                    va='center'
                )

        plt.title('Class Distribution Between Train/Test Splits', fontsize=16, pad=20)
        plt.ylabel('Class', fontsize=12)
        plt.tight_layout()

        plt.savefig(self.output_dir / 'class_train_test_split.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Get the path to the folder containing LAS files
    data_folder = r"C:\Users\s4824030\PycharmProjects\hfx3d-benchmark\data\WACV\processed"

    # Create analyzer and run analysis
    analyzer = TrainTestSplitAnalyzer(data_folder)
    results = analyzer.analyze_dataset()

    if results:
        analyzer.visualize_split(results)
    else:
        print("Analysis failed - check your input data")

if __name__ == "__main__":
    main()
