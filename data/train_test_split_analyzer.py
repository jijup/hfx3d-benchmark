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
        
    def load_label_map(self, json_path="label_map.json"):
        """Load the label map from JSON file."""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "unlabelled": 0, "wall": 1, "window": 2, "door": 3,
                "balcony": 4, "vegetation": 5, "stairs": 6, "terrain": 7,
                "roof": 8, "blinds": 9, "other": 10, "column": 11, "arch": 12
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
        
        # 1. Horizontal bar chart comparing train/test class distribution
        self._plot_class_distribution_comparison(analysis_results)
        
        # 2. Normalized distribution (percentage)
        self._plot_normalized_distribution(analysis_results)
        
        # 3. Class presence heatmap
        self._plot_class_presence_heatmap(analysis_results)
        
        # 4. New visualization showing train/test split ratio for each class
        self._plot_class_train_test_split_ratio(analysis_results)
        
        # 5. Generate detailed report
        self._generate_split_report(analysis_results)
        
    def _plot_class_distribution_comparison(self, results):
        """Create horizontal bar chart comparing train/test distributions."""
        train_counts = results['train_class_counts']
        test_counts = results['test_class_counts']
        
        # Get all class IDs
        all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
        
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
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Sort classes by total count (descending)
        class_order = df.groupby('Class')['Count'].sum().sort_values(ascending=False).index
        
        # Create horizontal bar chart
        palette_dict = {'Train': self.colors['train'], 'Test': self.colors['test']}
        
        sns.barplot(
            x='Count', 
            y='Class', 
            hue='Split',
            data=df,
            palette=palette_dict,
            order=class_order,
            orient='h'
        )
        
        # Add count labels on bars with abbreviated format
        def format_number(num):
            if num >= 1_000_000:
                return f'{num/1_000_000:.1f}M'
            elif num >= 1_000:
                return f'{num/1_000:.1f}K'
            else:
                return f'{num:.0f}'
        
        for i, p in enumerate(plt.gca().patches):
            width = p.get_width()
            plt.text(
                width + (width * 0.01),
                p.get_y() + p.get_height()/2,
                format_number(width),
                ha='left', 
                va='center'
            )
        
        # Format x-axis labels with appropriate scale
        ax = plt.gca()
        current_values = ax.get_xticks()
        ax.set_xticks(current_values)
        ax.set_xticklabels([format_number(x) for x in current_values])
        
        plt.title('Train vs Test Class Distribution', fontsize=16, pad=20)
        plt.xlabel('Number of Points', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a log-scale version for better visibility of smaller classes
        plt.figure(figsize=(14, 10))
        
        sns.barplot(
            x='LogCount', 
            y='Class', 
            hue='Split',
            data=df,
            palette=palette_dict,
            order=class_order,
            orient='h'
        )
        
        # Properly format log scale ticks with scientific notation
        ax = plt.gca()
        log_ticks = ax.get_xticks()
        ax.set_xticks(log_ticks)
        
        # Now set the formatted labels with scientific notation (10^x)
        ax.set_xticklabels([f'$10^{{{int(x)}}}$' if x.is_integer() else f'$10^{{{x:.1f}}}$' for x in log_ticks])
        
        plt.title('Train vs Test Class Distribution (Log Scale)', fontsize=16, pad=20)
        plt.xlabel('Number of Points (log scale)', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution_comparison_log.png', dpi=300, bbox_inches='tight')
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
            palette=palette_dict,  # Changed from self.colors to palette_dict
            order=class_order,
            orient='h'
        )
        
        # Add percentage labels on bars
        for i, p in enumerate(plt.gca().patches):
            width = p.get_width()
            if width > 0.5:  # Only add label if percentage is significant
                plt.text(
                    width + 0.1,
                    p.get_y() + p.get_height()/2,
                    f'{width:.1f}%',
                    ha='left', 
                    va='center'
                )
        
        plt.title('Train vs Test Class Distribution (Percentage)', fontsize=16, pad=20)
        plt.xlabel('Percentage of Points', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution_percentage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_class_presence_heatmap(self, results):
        """Create a heatmap showing class presence across all files."""
        per_file_counts = results['per_file_counts']
        
        # Get all classes and files
        all_classes = sorted(set(class_id for counts in per_file_counts.values() for class_id in counts.keys()))
        all_files = sorted(per_file_counts.keys())
        
        # Create a matrix of class presence (1) or absence (0)
        presence_matrix = np.zeros((len(all_files), len(all_classes)))
        
        for i, filename in enumerate(all_files):
            for j, class_id in enumerate(all_classes):
                if class_id in per_file_counts[filename]:
                    presence_matrix[i, j] = 1
        
        # Create a DataFrame
        df = pd.DataFrame(
            presence_matrix, 
            index=all_files,
            columns=[self.id_to_name.get(c, f'Unknown-{c}').capitalize() for c in all_classes]
        )
        
        # Create a list of filenames in sorted order (train files first, then test files)
        train_files = []
        test_files = []
        for filename in all_files:
            if self.is_train_file(filename):
                train_files.append(filename)
            else:
                test_files.append(filename)
        
        # Sort each group
        train_files.sort()
        test_files.sort()
        
        # Combine sorted lists
        sorted_files = train_files + test_files
        
        # Reindex DataFrame with sorted files
        df = df.reindex(sorted_files)
        
        # Create split indicators list in the same order
        split_indicators = []
        for filename in sorted_files:
            if self.is_train_file(filename):
                split_indicators.append('Train')
            else:
                split_indicators.append('Test')
        
        # Create figure
        plt.figure(figsize=(16, len(all_files) * 0.4 + 2))
        
        # Create heatmap
        ax = sns.heatmap(
            df, 
            cmap=['white', '#36c946'],  # White for 0, green for 1
            cbar=False,
            linewidths=0.5,
            linecolor='gray'
        )
        
        # Add colored bars to indicate train/test split
        for i, split in enumerate(split_indicators):
            if split == 'Train':
                color = self.colors['train']
            else:
                color = self.colors['test']
            ax.add_patch(plt.Rectangle((-0.5, i-0.5), 0.5, 1, color=color))
        
        # Custom legend for train/test
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['train'], label='Train'),
            Patch(facecolor=self.colors['test'], label='Test')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('Class Presence Across Files', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_presence_heatmap.png', dpi=300, bbox_inches='tight')
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
                    class_name = self.id_to_name.get(class_id, f'Unknown-{class_id}')  # FIXED: Changed 'c' to 'class_id'
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
    
        # This is outside the with block, which is fine
        print(f"Analysis complete! Results saved to {self.output_dir}")

    def _plot_class_train_test_split_ratio(self, results):
        """Create horizontal bar chart showing how each class is split between train/test."""
        train_counts = results['train_class_counts']
        test_counts = results['test_class_counts']
        
        # Get all class IDs
        all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
        
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
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Sort classes by total count (descending)
        class_order = df.groupby('Class')['Total'].max().sort_values(ascending=False).index
        
        # Create horizontal bar chart
        palette_dict = {'Train': self.colors['train'], 'Test': self.colors['test']}
        
        # Create the plot with logarithmic y-scale
        ax = plt.gca()
        
        sns.barplot(
            x='Percentage', 
            y='Class', 
            hue='Split',
            data=df,
            palette=palette_dict,
            order=class_order,
            orient='h',
            ax=ax
        )
        
        # Add percentage and count labels on bars
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            if width > 2:  # Only add label if percentage is significant (> 2%)
                row = df.iloc[i]
                plt.text(
                    width + 0.5,
                    p.get_y() + p.get_height()/2,
                    f'{width:.1f}% ({int(row["Count"]):,})',
                    ha='left', 
                    va='center'
                )
        
        plt.title('Class Distribution Between Train/Test Splits', fontsize=16, pad=20)
        plt.xlabel('Percentage of Class in Each Split', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.xlim(0, 105)  # Set x-axis limit to 105% to leave room for labels
        
        # Add a second axes for counts (on the right side)
        ax2 = ax.twinx()
        
        # Create log-scale y-axis for counts
        count_data = df.groupby('Class')['Total'].max().reindex(class_order)
        y_pos = np.arange(len(class_order))
        
        # Set up logarithmic scale starting from 10^2
        ax2.set_yscale('log')
        ax2.set_ylim(100, max(count_data) * 1.5)  # Start at 10^2
        
        # Minimal styling for right axis
        ax2.grid(False)
        ax2.set_ylabel('Total Points (log scale)', fontsize=12)
        
        # Plot invisible points to make the scale work with our ordered classes
        ax2.scatter(np.zeros_like(y_pos), count_data.values, alpha=0)
        
        # Format the tick labels
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_train_test_split.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Get the path to the folder containing LAS files
    data_folder = r"C:\Users\abdia\PycharmProjects\hfx3d-benchmark\data\WACV"
    
    # Create analyzer and run analysis
    analyzer = TrainTestSplitAnalyzer(data_folder)
    results = analyzer.analyze_dataset()
    
    if results:
        analyzer.visualize_split(results)
    else:
        print("Analysis failed - check your input data")

if __name__ == "__main__":
    main()