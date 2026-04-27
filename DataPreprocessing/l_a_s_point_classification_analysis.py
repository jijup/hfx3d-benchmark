import os
import laspy
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tk/Tcl issues
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy.stats import entropy


class PointCloudAnalyzer:
    def __init__(self, output_dir=None):
        self.label_map = self.load_label_map()
        self.id_to_name = {v: k for k, v in self.label_map.items()}
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")
        os.makedirs(self.output_dir, exist_ok=True)

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_label_map(self, json_path=r"C:\Users\s4824030\PycharmProjects\hfx3d-benchmark\configs\label_map.json"):
        with open(json_path, 'r') as f:
            return json.load(f)

    def analyze_directory(self, directory):
        directory = Path(directory)
        las_files = list(directory.glob("*.las"))

        if not las_files:
            print("No .las files found!")
            return None

        results = {}
        total_stats = defaultdict(int)

        for las_file in las_files:
            print(f"Processing: {las_file.name}")
            try:
                las_data = laspy.read(las_file)
                class_counts = dict(zip(*np.unique(las_data.classification, return_counts=True)))
                results[las_file.name] = class_counts
                for class_id, count in class_counts.items():
                    total_stats[class_id] += count
            except Exception as e:
                print(f"Error processing {las_file.name}: {str(e)}")

        return {'per_file': results, 'total_stats': dict(total_stats)}

    def generate_visualizations(self, analysis_results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._plot_overall_distribution(analysis_results['total_stats'], timestamp)
        self._plot_class_heatmap(analysis_results['per_file'], timestamp)
        self._plot_class_balance_metrics(analysis_results['total_stats'], timestamp)
        self._plot_file_statistics(analysis_results['per_file'], timestamp)
        self._generate_summary_report(analysis_results, timestamp)

    def _plot_overall_distribution(self, total_stats, timestamp):
        sorted_items = sorted(total_stats.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_items)

        fig, ax1 = plt.subplots(figsize=(15, 8))

        bars = ax1.bar(range(len(classes)), counts, color='skyblue', alpha=0.7)
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.YlOrRd(np.exp(-i/5)))
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)

        ax1.set_title('Long-Tail Distribution of Point Cloud Classes', fontsize=16, pad=20)
        ax1.set_xlabel('Class Name', fontsize=14)
        ax1.set_ylabel('Number of Points (log scale)', fontsize=14)
        ax1.set_yscale('log')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels([self.id_to_name.get(c, f'Unknown-{c}').capitalize() for c in classes],
                             rotation=45, ha='right', fontsize=10)

        total = sum(counts)
        cumulative_pct = np.cumsum(counts) / total * 100

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height * 1.05, f'{int(height):,}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

        eighty_pct_idx = np.where(cumulative_pct >= 80)[0][0]
        ax1.axvline(x=eighty_pct_idx, color='black', linestyle='--', alpha=0.7)
        ax1.text(eighty_pct_idx + 0.5, ax1.get_ylim()[0] * 10,
                 f'80% of points\nin {eighty_pct_idx+1} classes\n({(eighty_pct_idx+1)/len(classes):.1%} of classes)',
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        ax1.grid(True, axis='y', alpha=0.3)

        info_text = (f"Dataset Statistics:\n"
                     f"• Total points: {total:,}\n"
                     f"• Classes: {len(classes)}\n"
                     f"• Top class ({self.id_to_name.get(classes[0], 'Unknown')}) has {counts[0]/total:.1%} of points")
        
        plt.figtext(0.01, 0.01, info_text, fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)  # Make room for x-labels and stats annotation
        plt.savefig(os.path.join(self.output_dir, f'class_distribution_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_class_heatmap(self, per_file_results, timestamp):
        classes = sorted(set(class_id for results in per_file_results.values()
                             for class_id in results.keys()))
        files = list(per_file_results.keys())

        data = np.zeros((len(files), len(classes)))
        for i, file in enumerate(files):
            for j, class_id in enumerate(classes):
                data[i, j] = per_file_results[file].get(class_id, 0)

        plt.figure(figsize=(15, 10))
        sns.heatmap(data,
                    xticklabels=[self.id_to_name.get(c, f'Unknown-{c}') for c in classes],
                    yticklabels=[f.replace('.las', '') for f in files],
                    cmap='YlOrRd',
                    norm='log')

        plt.title('Point Distribution Heatmap (log scale)', fontsize=14, pad=20)
        plt.xlabel('Class Name', fontsize=12)
        plt.ylabel('Building', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'distribution_heatmap_{timestamp}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    def _plot_class_balance_metrics(self, total_stats, timestamp):
        total_points = sum(total_stats.values())
        proportions = {k: v / total_points for k, v in total_stats.items()}

        plt.figure(figsize=(26, 10))

        labels_pie = [self.id_to_name.get(k, f'Unknown-{k}').capitalize() for k in proportions.keys()]
        values_pie = list(proportions.values())

        ax_pie = plt.subplot(1, 2, 1)
        wedges, _ = ax_pie.pie(
            values_pie,
            labels=None,
            autopct=None,
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=0.5)
        )
        ax_pie.set_title('Class Proportions', pad=20)

        SMALL_THR = 0.02  # slices < 2% get fanned-out arrow labels
        small_right, small_left = [], []
        for wedge, name, val in zip(wedges, labels_pie, values_pie):
            mid_rad = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            cx, cy = np.cos(mid_rad), np.sin(mid_rad)
            if val >= SMALL_THR:
                ha = 'left' if cx >= 0 else 'right'
                ax_pie.text(cx * 1.15, cy * 1.15, f"{name}\n{val*100:.1f}%",
                            ha=ha, va='center', fontsize=8)
            else:
                (small_right if cx >= 0 else small_left).append((cx, cy, name, val))

        for items, x_lbl, ha in [(small_right, 1.65, 'left'), (small_left, -1.65, 'right')]:
            if not items:
                continue
            ys = np.linspace(0.55, -0.55, len(items))
            for (cx, cy, name, val), y_pos in zip(items, ys):
                ax_pie.annotate(f"{name}: {val*100:.1f}%",
                                xy=(cx * 0.92, cy * 0.92),
                                xytext=(x_lbl, y_pos),
                                ha=ha, va='center', fontsize=7,
                                arrowprops=dict(arrowstyle='-', color='gray',
                                                lw=0.6, shrinkA=0, shrinkB=2))
        ax_pie.set_xlim(-2.2, 2.2)

        plt.subplot(1, 2, 2)
        sorted_items = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
        classes, props = zip(*sorted_items)
        counts = [p * total_points for p in props]

        ax1 = plt.gca()
        bars = ax1.bar(range(len(classes)), counts, alpha=0.7, width=0.7)
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.YlOrRd(np.exp(-i/3)))
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)

        ax1.set_yscale('log')
        ax1.set_xlabel('Class Name', fontsize=12)
        ax1.set_ylabel('Number of Points (log scale)', fontsize=12)

        bar_heights = [bar.get_height() for bar in bars]
        ax1.plot(range(len(classes)), bar_heights, 'r-', linewidth=2, marker='o', markersize=5)

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05, f'{int(height):,}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.xticks(range(len(classes)),
                   [self.id_to_name.get(c, f'Unknown-{c}').capitalize() for c in classes],
                   rotation=45, ha='right', fontsize=9)
        plt.title('Class Distribution', fontsize=12)

        eighty_pct_idx = np.where(np.cumsum(counts) / total_points >= 0.8)[0][0]
        ax1.axvline(x=eighty_pct_idx, color='black', linestyle='--', alpha=0.7)
        ax1.text(eighty_pct_idx + 0.3, bar_heights[0] * 0.1,
                 f'80% of points in\n{eighty_pct_idx+1} classes',
                 fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout(w_pad=4)
        plt.subplots_adjust(bottom=0.35)
        plt.savefig(os.path.join(self.output_dir, f'class_balance_metrics_{timestamp}.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.5)
        plt.close()

    def _plot_file_statistics(self, per_file_results, timestamp):
        labels = [f.replace('.las', '') for f in per_file_results.keys()]
        total_points = [sum(c.values()) for c in per_file_results.values()]
        num_classes = [len(c) for c in per_file_results.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.bar(range(len(labels)), total_points, color='steelblue', alpha=0.8)
        ax1.set_title('Total Points per Building')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Total Points (log scale)')
        ax1.set_yscale('log')

        ax2.bar(range(len(labels)), num_classes, color='steelblue', alpha=0.8)
        ax2.set_title('Number of Classes per Building')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Number of Classes')

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

            f.write("Overall Statistics:\n")
            f.write(f"Total number of files: {len(per_file_results)}\n")
            f.write(f"Total number of points: {total_points:,}\n")
            f.write(f"Number of classes: {len(total_stats)}\n\n")

            f.write("Class Distribution:\n")
            for class_id, count in sorted(total_stats.items(), key=lambda x: x[1], reverse=True):
                class_name = self.id_to_name.get(class_id, f'Unknown-{class_id}')
                percentage = (count / total_points) * 100
                f.write(f"{class_name:<15} {count:>12,} points ({percentage:>6.2f}%)\n")

            proportions = [count / total_points for count in total_stats.values()]
            class_entropy = entropy(proportions)
            gini = 1 - sum(p * p for p in proportions)
            f.write("\nImbalance Metrics:\n")
            f.write(f"Entropy: {class_entropy:.4f}\n")
            f.write(f"Gini coefficient: {gini:.4f}\n")

            f.write("\n\nPer-Building Class Distribution:\n")
            f.write("=" * 50 + "\n")

            class_ids = sorted(set(class_id for results in per_file_results.values()
                                   for class_id in results.keys()))
            class_names = [self.id_to_name.get(class_id, f'Unknown-{class_id}') for class_id in class_ids]

            f.write(f"{'Building':<20} {'Total Points':>12}")
            for class_name in class_names:
                f.write(f" | {class_name:>10}")
            f.write("\n")
            f.write("-" * 20 + " " + "-" * 12 + " " + "-" * (14 * len(class_names)) + "\n")

            for building, counts in per_file_results.items():
                building_total = sum(counts.values())
                f.write(f"{building:<20} {building_total:>12,}")
                for class_id in class_ids:
                    f.write(f" | {counts.get(class_id, 0):>10,}")
                f.write("\n")

            # Class presence matrix — useful for train/test split planning
            f.write("\n\nClass Presence Matrix (1=present, 0=absent):\n")
            f.write("=" * 50 + "\n")

            f.write(f"{'Building':<20}")
            for class_name in class_names:
                f.write(f" | {class_name:>5}")
            f.write("\n")
            f.write("-" * 20 + " " + "-" * (9 * len(class_names)) + "\n")

            for building, counts in per_file_results.items():
                f.write(f"{building:<20}")
                for class_id in class_ids:
                    f.write(f" | {1 if class_id in counts else 0:>5}")
                f.write("\n")

def main():
    analyzer = PointCloudAnalyzer()
    dir_path = r"C:\Users\s4824030\PycharmProjects\hfx3d-benchmark\data\WACV"
    results = analyzer.analyze_directory(dir_path)
    if results:
        analyzer.generate_visualizations(results)
        print(f"\nAnalysis complete! Results saved to: {analyzer.output_dir}")
        os.startfile(analyzer.output_dir)


if __name__ == "__main__":
    main()
