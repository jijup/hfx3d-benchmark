import os
import laspy
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
from scipy.stats import entropy


class PointCloudAnalyzer:
    def __init__(self, output_dir=None):
        self.label_map = self.load_label_map()
        self.id_to_name = {v: k for k, v in self.label_map.items()}
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_label_map(self, json_path=r"C:\Users\s4824030\PycharmProjects\hfx3d-benchmark\configs\label_map.json"):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "wall": 0, "window": 1, "door": 2,
                "balcony": 3, "vegetation": 4, "stairs": 5, "terrain": 6,
                "roof": 7, "blinds": 8, "other": 9, "column": 10, "arch": 11
            }

    def analyze_directory(self, directory):
        """Analyze all .las files in directory"""
        directory = Path(directory)
        las_files = list(directory.glob("*.las"))

        if not las_files:
            print("No .las files found!")
            return None

        results = {}
        total_stats = defaultdict(int)
        file_stats = []

        for las_file in las_files:
            print(f"Processing: {las_file.name}")
            try:
                las_data = laspy.read(las_file)
                class_counts = dict(zip(*np.unique(las_data.classification, return_counts=True)))
                results[las_file.name] = class_counts

                # Update total statistics
                for class_id, count in class_counts.items():
                    total_stats[class_id] += count

                # Store per-file statistics
                file_stats.append({
                    'filename': las_file.name,
                    'total_points': sum(class_counts.values()),
                    'class_counts': class_counts,
                    'num_classes': len(class_counts)
                })

            except Exception as e:
                print(f"Error processing {las_file.name}: {str(e)}")

        return {
            'per_file': results,
            'total_stats': dict(total_stats),
            'file_stats': file_stats
        }

    def generate_visualizations(self, analysis_results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Overall Class Distribution
        self._plot_overall_distribution(analysis_results['total_stats'], timestamp)

        # 2. Class Distribution Heatmap
        self._plot_class_heatmap(analysis_results['per_file'], timestamp)

        # 3. Class Balance Analysis
        self._plot_class_balance_metrics(analysis_results['total_stats'], timestamp)

        # 4. Per-File Statistics
        self._plot_file_statistics(analysis_results['file_stats'], timestamp)

        # 5. Generate Summary Report
        self._generate_summary_report(analysis_results, timestamp)

    def _plot_overall_distribution(self, total_stats, timestamp):
        plt.figure(figsize=(15, 10))
        
        # Sort by count
        sorted_items = sorted(total_stats.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_items)
        
        # Create a figure with two subplots sharing the same x-axis
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        # Primary axis - bar chart
        bars = ax1.bar(range(len(classes)), counts, color='skyblue', alpha=0.7)
        
        # Add a gradient overlay/spotlight effect on the bars
        for i, bar in enumerate(bars):
            # Create a custom gradient color that highlights the first few classes
            # and gradually fades for the long tail
            gradient_factor = np.exp(-i/5)  # Exponential decay for spotlight effect
            bar.set_color(plt.cm.YlOrRd(gradient_factor))
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
        
        # Customize primary axis
        ax1.set_title('Long-Tail Distribution of Point Cloud Classes', fontsize=16, pad=20)
        ax1.set_xlabel('Class Name', fontsize=14)
        ax1.set_ylabel('Number of Points (log scale)', fontsize=14)
        ax1.set_yscale('log')
        
        # Rotate x-axis labels
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels([self.id_to_name.get(c, f'Unknown-{c}').capitalize() for c in classes],
                             rotation=45, ha='right', fontsize=10)
        
        # Secondary axis - cumulative percentage
        ax2 = ax1.twinx()
        total = sum(counts)
        cumulative_pct = np.cumsum(counts) / total * 100
        ax2.plot(range(len(classes)), cumulative_pct, 'ro-', linewidth=3, 
                 label='Cumulative %')
        ax2.set_ylabel('Cumulative Percentage', fontsize=14, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0, 105])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height * 1.05,
                     f'{int(height):,}',
                     ha='center', va='bottom', rotation=0,
                     fontsize=8, fontweight='bold')
        
        # Add percentage labels on the cumulative curve
        for i, (x, y) in enumerate(zip(range(len(classes)), cumulative_pct)):
            # Only show labels at interesting points (20%, 50%, 80%, 95%)
            if i == 0 or y >= 20 and cumulative_pct[i-1] < 20 or \
               y >= 50 and cumulative_pct[i-1] < 50 or \
               y >= 80 and cumulative_pct[i-1] < 80 or \
               y >= 95 and cumulative_pct[i-1] < 95:
                ax2.annotate(f'{y:.1f}%', 
                             xy=(x, y),
                             xytext=(5, 0), 
                             textcoords='offset points',
                             color='darkred',
                             fontweight='bold')
        
        # Highlight the Pareto principle (80/20 rule)
        eighty_pct_idx = np.where(cumulative_pct >= 80)[0][0]
        ax1.axvline(x=eighty_pct_idx, color='black', linestyle='--', alpha=0.7)
        ax1.text(eighty_pct_idx + 0.5, ax1.get_ylim()[0] * 10, 
                 f'80% of points\nin {eighty_pct_idx+1} classes\n({(eighty_pct_idx+1)/len(classes):.1%} of classes)',
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add a legend for the cumulative curve
        ax2.legend(loc='upper left')
        
        # Add grid for easier reading
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add annotations about the distribution
        props = counts / total
        entropy_val = entropy(props)
        gini = 1 - sum(p * p for p in props)
        
        info_text = (f"Dataset Statistics:\n"
                     f"• Total points: {total:,}\n"
                     f"• Classes: {len(classes)}\n"
                     f"• Class entropy: {entropy_val:.2f}\n"
                     f"• Gini coefficient: {gini:.2f}\n"
                     f"• Top class ({self.id_to_name.get(classes[0], 'Unknown')}) has {counts[0]/total:.1%} of points")
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for the annotations
        plt.savefig(os.path.join(self.output_dir, f'class_distribution_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_class_heatmap(self, per_file_results, timestamp):
        # Prepare data for heatmap
        classes = sorted(set(class_id for results in per_file_results.values()
                             for class_id in results.keys()))
        files = list(per_file_results.keys())

        data = np.zeros((len(files), len(classes)))
        for i, file in enumerate(files):
            for j, class_id in enumerate(classes):
                data[i, j] = per_file_results[file].get(class_id, 0)

        # Create heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(data,
                    xticklabels=[self.id_to_name.get(c, f'Unknown-{c}') for c in classes],
                    yticklabels=files,
                    cmap='YlOrRd',
                    norm='log')

        plt.title('Point Distribution Heatmap (log scale)', fontsize=14, pad=20)
        plt.xlabel('Class Name', fontsize=12)
        plt.ylabel('File Name', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'distribution_heatmap_{timestamp}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    def _plot_class_balance_metrics(self, total_stats, timestamp):
        # Calculate imbalance metrics
        total_points = sum(total_stats.values())
        proportions = {k: v / total_points for k, v in total_stats.items()}

        # Calculate entropy (higher means more balanced)
        class_entropy = entropy(list(proportions.values()))

        # Create visualization with increased size and spacing
        plt.figure(figsize=(22, 10))  # Increased figure size for better labels

        # Custom autopct function to only show percentages above 1%
        def make_autopct(values):
            def my_autopct(pct):
                if pct < 1:
                    return ''
                else:
                    return f'{pct:.1f}%'
            return my_autopct

        # Plot proportions with adjusted layout
        plt.subplot(1, 2, 1)
        wedges, texts, autotexts = plt.pie(
            proportions.values(),
            labels=[self.id_to_name.get(k, f'Unknown-{k}') for k in proportions.keys()],
            autopct=make_autopct(proportions.values()),
            pctdistance=0.85,
            labeldistance=1.1
        )
        # Make percentage labels smaller
        plt.setp(autotexts, size=8)
        # Make label texts smaller
        plt.setp(texts, size=8)
        plt.title('Class Proportions', pad=20)

        # Right subplot - Bar chart with connecting line
        plt.subplot(1, 2, 2)
        
        # Sort classes by proportion
        sorted_items = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
        classes, props = zip(*sorted_items)
        counts = [p * total_points for p in props]  # Convert back to counts
        class_names = [self.id_to_name.get(c, f'Unknown-{c}').capitalize() for c in classes]
        
        # Plot bars with gradient effect
        ax1 = plt.gca()
        bars = ax1.bar(range(len(classes)), counts, alpha=0.7, width=0.7)
        
        # Add gradient effect (spotlight) on bars
        for i, bar in enumerate(bars):
            gradient_factor = np.exp(-i/3)  # Spotlight effect
            bar.set_color(plt.cm.YlOrRd(gradient_factor))
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
        
        # Set log scale
        ax1.set_yscale('log')
        ax1.set_xlabel('Class Name', fontsize=12)
        ax1.set_ylabel('Number of Points (log scale)', fontsize=12)
        
        # Add a line connecting the tops of the bars
        bar_heights = [bar.get_height() for bar in bars]
        ax1.plot(range(len(classes)), bar_heights, 'r-', linewidth=2, marker='o', markersize=5)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                     f'{int(height):,}',
                     ha='center', va='bottom', rotation=0,
                     fontsize=8, fontweight='bold')
        
        # Custom x-axis labels with full class names
        plt.xticks(range(len(classes)),
                   [self.id_to_name.get(c, f'Unknown-{c}').capitalize() for c in classes],
                   rotation=45,
                   ha='right',
                   fontsize=9)

        plt.title(f'Class Distribution\nEntropy: {class_entropy:.2f}', fontsize=12)
        
        # Add Pareto principle line (80/20 rule)
        eighty_pct_idx = np.where(np.cumsum(counts) / total_points >= 0.8)[0][0]
        ax1.axvline(x=eighty_pct_idx, color='black', linestyle='--', alpha=0.7)
        ax1.text(eighty_pct_idx + 0.3, bar_heights[0] * 0.1,
             f'80% of points in\n{eighty_pct_idx+1} classes',
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        # Adjust layout with more spacing
        plt.tight_layout(w_pad=4)
        plt.subplots_adjust(bottom=0.25)  # Extra space at bottom for class names
        plt.savefig(os.path.join(self.output_dir, f'class_balance_metrics_{timestamp}.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.5)
        plt.close()

    def _plot_file_statistics(self, file_stats, timestamp):
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(file_stats)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot total points per file
        sns.barplot(data=df, x='filename', y='total_points', ax=ax1)
        ax1.set_title('Total Points per File')
        # Fix for tick labels
        ax1.set_xticks(range(len(df['filename'])))
        ax1.set_xticklabels(df['filename'], rotation=45, ha='right')
        ax1.set_yscale('log')

        # Plot number of classes per file
        sns.barplot(data=df, x='filename', y='num_classes', ax=ax2)
        ax2.set_title('Number of Classes per File')
        # Fix for tick labels
        ax2.set_xticks(range(len(df['filename'])))
        ax2.set_xticklabels(df['filename'], rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'file_statistics_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_summary_report(self, analysis_results, timestamp):
        total_stats = analysis_results['total_stats']
        total_points = sum(total_stats.values())
        per_file_results = analysis_results['per_file']

        with open(os.path.join(self.output_dir, f'analysis_summary_{timestamp}.txt'), 'w') as f:
            f.write("Point Cloud Dataset Analysis Summary\n")
            f.write("=" * 50 + "\n\n")

            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"Total number of files: {len(per_file_results)}\n")
            f.write(f"Total number of points: {total_points:,}\n")
            f.write(f"Number of classes: {len(total_stats)}\n\n")

            # Class distribution
            f.write("Class Distribution:\n")
            for class_id, count in sorted(total_stats.items(), key=lambda x: x[1], reverse=True):
                class_name = self.id_to_name.get(class_id, f'Unknown-{class_id}')
                percentage = (count / total_points) * 100
                f.write(f"{class_name:<15} {count:>12,} points ({percentage:>6.2f}%)\n")

            # Calculate imbalance metrics
            proportions = [count / total_points for count in total_stats.values()]
            class_entropy = entropy(proportions)
            gini = 1 - sum(p * p for p in proportions)

            f.write("\nImbalance Metrics:\n")
            f.write(f"Entropy: {class_entropy:.4f}\n")
            f.write(f"Gini coefficient: {gini:.4f}\n")
            
            # Per-building statistics
            f.write("\n\nPer-Building Class Distribution:\n")
            f.write("=" * 50 + "\n")
            
            # Create a table header with class names
            class_ids = sorted(set(class_id for results in per_file_results.values() 
                                  for class_id in results.keys()))
            class_names = [self.id_to_name.get(class_id, f'Unknown-{class_id}') for class_id in class_ids]
            
            # Table header
            f.write(f"{'Building':<20} {'Total Points':>12}")
            for class_name in class_names:
                f.write(f" | {class_name:>10}")
            f.write("\n")
            f.write("-" * 20 + " " + "-" * 12 + " " + "-" * (14 * len(class_names)) + "\n")
            
            # Data rows
            for building, counts in per_file_results.items():
                building_total = sum(counts.values())
                f.write(f"{building:<20} {building_total:>12,}")
                
                for class_id in class_ids:
                    count = counts.get(class_id, 0)
                    f.write(f" | {count:>10,}")
                f.write("\n")
            
            # Class presence matrix (useful for train/test split planning)
            f.write("\n\nClass Presence Matrix (1=present, 0=absent):\n")
            f.write("=" * 50 + "\n")
            
            # Table header
            f.write(f"{'Building':<20}")
            for class_name in class_names:
                f.write(f" | {class_name:>5}")
            f.write("\n")
            f.write("-" * 20 + " " + "-" * (9 * len(class_names)) + "\n")
            
            # Data rows
            for building, counts in per_file_results.items():
                f.write(f"{building:<20}")
                
                for class_id in class_ids:
                    present = 1 if class_id in counts else 0
                    f.write(f" | {present:>5}")
                f.write("\n")

    def plot_advanced_class_distribution(self, total_stats, timestamp):
        # Sort classes by count (descending)
        sorted_items = sorted(total_stats.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_items)
        class_names = [self.id_to_name.get(c, f'Unknown-{c}') for c in classes]
        
        # Create figure with multiple panels
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Colored histogram with custom color for each class
        ax1 = plt.subplot(2, 2, 1)
        # Use a colormap that's visually distinct
        colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
        bars = ax1.bar(range(len(classes)), counts, color=colors)
        
        # Customize plot
        ax1.set_title('Class Distribution (log scale)', fontsize=14)
        ax1.set_xlabel('Class Name', fontsize=12)
        ax1.set_ylabel('Number of Points', fontsize=12)
        ax1.set_yscale('log')
        
        # Rotate x-axis labels
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height):,}',
                     ha='center', va='bottom', rotation=0, fontsize=8)
        
        # 2. Long Tail Analysis - Power Law Plot
        ax2 = plt.subplot(2, 2, 2)
        # Log-log plot to identify power law distribution
        ax2.loglog(range(1, len(counts)+1), counts, 'ro-', markersize=8)
        ax2.set_title('Long Tail Distribution (log-log)', fontsize=14)
        ax2.set_xlabel('Rank', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        
        # Fit power law to identify the exponent (if it follows a power law)
        from scipy import stats
        ranks = np.arange(1, len(counts)+1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log(ranks), np.log(counts)
        )
        ax2.plot(ranks, np.exp(intercept) * ranks**slope, 'b--', 
                 label=f'Power law fit (α={-slope:.2f})')
        ax2.legend()
        
        # 3. Pareto Analysis (cumulative distribution)
        ax3 = plt.subplot(2, 2, 3)
        total = sum(counts)
        cumulative_pct = np.cumsum(counts) / total * 100
        
        # Primary Y-axis: Class counts
        bars = ax3.bar(range(len(classes)), counts, color=colors, alpha=0.7)
        ax3.set_xlabel('Classes (sorted by frequency)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        
        # Secondary Y-axis: Cumulative percentage
        ax3_twin = ax3.twinx()
        ax3_twin.plot(range(len(classes)), cumulative_pct, 'ro-', linewidth=2)
        ax3_twin.set_ylabel('Cumulative %', fontsize=12, color='r')
        ax3_twin.tick_params(axis='y', colors='r')
        ax3_twin.set_ylim([0, 105])
        
        # Mark the 80% threshold (Pareto principle)
        eighty_pct_idx = np.where(cumulative_pct >= 80)[0][0]
        ax3.axvline(x=eighty_pct_idx, color='k', linestyle='--')
        ax3.text(eighty_pct_idx+0.5, max(counts)/2, 
                 f'{eighty_pct_idx+1} classes\n= 80% of data', 
                 fontsize=10)
        
        ax3.set_title('Pareto Analysis', fontsize=14)
        
        # 4. Head-to-Tail Ratio Analysis
        ax4 = plt.subplot(2, 2, 4)
        
        # Define head (top 20% classes) and tail (bottom 80% classes)
        head_count = int(np.ceil(len(classes) * 0.2))
        head_classes = classes[:head_count]
        tail_classes = classes[head_count:]
        
        head_total = sum(counts[:head_count])
        tail_total = sum(counts[head_count:])
        
        # Create pie chart of head vs. tail
        ax4.pie([head_total, tail_total], 
                labels=[f'Head\n({head_count} classes)', f'Tail\n({len(tail_classes)} classes)'],
                autopct='%1.1f%%', 
                colors=['#ff9999','#66b3ff'],
                explode=(0.1, 0), 
                shadow=True)
        
        head_tail_ratio = head_total / tail_total if tail_total > 0 else float('inf')
        ax4.set_title(f'Head-to-Tail Ratio: {head_tail_ratio:.1f}', fontsize=14)
        
        # Add overall information
        plt.figtext(0.5, 0.01, 
                    f"Total classes: {len(classes)} | Total points: {total:,} | " +
                    f"Top 3 classes: {', '.join(class_names[:3])} | " +
                    f"Bottom 3 classes: {', '.join(class_names[-3:])}", 
                    ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(self.output_dir, f'advanced_distribution_{timestamp}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def main():
    analyzer = PointCloudAnalyzer()

    # Get directory path from user
    dir_path = r"C:\Users\s4824030\PycharmProjects\hfx3d-benchmark\data\WACV\processed"

    # Analyze directory
    results = analyzer.analyze_directory(dir_path)

    if results:
        # Generate visualizations and report
        analyzer.generate_visualizations(results)
        print(f"\nAnalysis complete! Results have been saved to: {analyzer.output_dir}")

        # Try to open the output directory
        try:
            os.startfile(analyzer.output_dir)  # Windows
        except AttributeError:
            try:
                import subprocess
                subprocess.call(['xdg-open', analyzer.output_dir])  # Linux
            except:
                try:
                    subprocess.call(['open', analyzer.output_dir])  # macOS
                except:
                    print(f"Could not open output directory automatically. Please check: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
