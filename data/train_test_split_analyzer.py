import os
import argparse
import shutil
import laspy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.palette import SPLIT as _SPLIT, CHART_STYLE, CHART_RC


class TrainTestSplitAnalyzer:
    def __init__(self, data_folder, output_dir=None):
        self.data_folder = Path(data_folder)
        self.output_dir = Path(output_dir) if output_dir else Path("split_analysis_output")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.train_buildings = self._load_split_file('train')
        self.val_buildings   = self._load_split_file('val')
        self.test_buildings  = self._load_split_file('test')

        self.label_map = self.load_label_map()
        self.id_to_name = {v: k for k, v in self.label_map.items()}

        plt.style.use(CHART_STYLE)
        plt.rcParams.update(CHART_RC)
        sns.set_context("notebook", font_scale=1.2)
        self.colors = {
            'train': _SPLIT['train'],
            'val':   _SPLIT['val'],
            'test':  _SPLIT['test'],
        }

    def _load_split_file(self, split_name):
        txt = Path(__file__).resolve().parent / 'splits' / f'{split_name}.txt'
        if not txt.exists():
            raise FileNotFoundError(f"Split file not found: {txt}")
        return [ln.strip() for ln in txt.read_text().splitlines()
                if ln.strip() and not ln.startswith('#')]

    def format_number(self, num):
        if num == 0:
            return "0"
        elif num < 1000:
            return f"{int(num)}"
        elif num < 1_000_000:
            return f"{num/1000:.1f}K"
        elif num < 1_000_000_000:
            return f"{num/1_000_000:.1f}M"
        else:
            return f"{num/1_000_000_000:.1f}B"

    def load_label_map(self, json_path=None):
        if json_path is None:
            json_path = Path(__file__).resolve().parent.parent / 'configs' / 'label_map.json'
        with open(json_path, 'r') as f:
            return json.load(f)

    def _match_building(self, filename, building_list):
        stem = Path(filename).stem.upper()
        return any(bld.upper() in stem for bld in building_list)

    def is_train_file(self, filename):
        return self._match_building(filename, self.train_buildings)

    def is_val_file(self, filename):
        return self._match_building(filename, self.val_buildings)

    def is_test_file(self, filename):
        return self._match_building(filename, self.test_buildings)

    def analyze_dataset(self):
        las_files = list(self.data_folder.glob("*.las"))
        if not las_files:
            print(f"No .las files found in {self.data_folder}")
            return None

        train_files, val_files, test_files = [], [], []
        train_counts = defaultdict(int)
        val_counts   = defaultdict(int)
        test_counts  = defaultdict(int)
        per_file_counts = {}

        for las_path in las_files:
            filename = las_path.name
            split = None
            if self.is_train_file(filename):
                split = 'train'
            elif self.is_val_file(filename):
                split = 'val'
            elif self.is_test_file(filename):
                split = 'test'
            else:
                print(f"Skipping {filename} — doesn't match any split")
                continue

            print(f"Processing [{split}]: {filename}")
            try:
                las_data = laspy.read(las_path)
                class_counts = dict(Counter(las_data.classification))
                per_file_counts[filename] = {'split': split, 'counts': class_counts}

                target = {'train': train_counts, 'val': val_counts, 'test': test_counts}[split]
                for class_id, count in class_counts.items():
                    target[class_id] += count

                {'train': train_files, 'val': val_files, 'test': test_files}[split].append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return {
            'train_files': train_files,
            'val_files':   val_files,
            'test_files':  test_files,
            'train_class_counts': dict(train_counts),
            'val_class_counts':   dict(val_counts),
            'test_class_counts':  dict(test_counts),
            'per_file_counts':    per_file_counts,
        }

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def visualize_split(self, analysis_results):
        if not analysis_results:
            print("No analysis results to visualize")
            return
        self._plot_class_distribution_comparison(analysis_results)
        self._generate_split_report(analysis_results)

    def _plot_class_distribution_comparison(self, results):
        """Horizontal bar chart — absolute point counts (log scale) per split."""
        train_counts = results['train_class_counts']
        val_counts   = results['val_class_counts']
        test_counts  = results['test_class_counts']

        all_classes = sorted(set(
            list(train_counts.keys()) + list(val_counts.keys()) + list(test_counts.keys())
        ))

        palette_dict = {'Train': self.colors['train'], 'Val': self.colors['val'], 'Test': self.colors['test']}

        data = []
        for class_id in all_classes:
            class_name = self.id_to_name.get(class_id, f'Unknown-{class_id}').capitalize()
            for split_label, counts in [('Train', train_counts), ('Val', val_counts), ('Test', test_counts)]:
                data.append({
                    'Class': class_name,
                    'Split': split_label,
                    'Count': counts.get(class_id, 0),
                })

        df = pd.DataFrame(data)
        class_order = df.groupby('Class')['Count'].sum().sort_values(ascending=False).index

        plt.figure(figsize=(16, 12))
        ax = plt.gca()
        sns.barplot(x='Count', y='Class', hue='Split', data=df,
                    palette=palette_dict, order=class_order, orient='h')

        ax.set_xscale('log')
        ax.set_xlim(100_000, None)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: self.format_number(x)))

        for p in ax.patches:
            w = p.get_width()
            if w >= 100_000:
                ax.text(w * 1.05, p.get_y() + p.get_height() / 2,
                        self.format_number(w), ha='left', va='center', fontsize=9)

        plt.title('Train / Val / Test Splits — HFX3D', fontsize=16, pad=20)
        plt.xlabel('Number of Points (log scale)', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
        plt.savefig(self.output_dir / 'Train_Val_Test_Splits_HFX3D.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: Train_Val_Test_Splits_HFX3D.png")

    def _generate_split_report(self, results):
        train_counts = results['train_class_counts']
        val_counts   = results['val_class_counts']
        test_counts  = results['test_class_counts']

        train_total = sum(train_counts.values())
        val_total   = sum(val_counts.values())
        test_total  = sum(test_counts.values())
        overall     = train_total + val_total + test_total

        all_classes = sorted(set(
            list(train_counts.keys()) + list(val_counts.keys()) + list(test_counts.keys())
        ))

        with open(self.output_dir / 'split_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("Train / Val / Test Split Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("File Distribution:\n")
            f.write(f"  Train ({len(results['train_files'])}): {', '.join(results['train_files'])}\n")
            f.write(f"  Val   ({len(results['val_files'])}):   {', '.join(results['val_files'])}\n")
            f.write(f"  Test  ({len(results['test_files'])}):  {', '.join(results['test_files'])}\n\n")

            f.write("Point Distribution:\n")
            for label, total in [('Train', train_total), ('Val', val_total), ('Test', test_total)]:
                f.write(f"  {label:<6}: {total:>15,}  ({total/overall:.1%} of total)\n")
            f.write(f"  {'Total':<6}: {overall:>15,}\n\n")

            f.write(f"{'Class':<15} {'Train':>14} {'Val':>14} {'Test':>14} {'Total':>14}\n")
            f.write("-" * 75 + "\n")
            for class_id in all_classes:
                name  = self.id_to_name.get(class_id, f'Unknown-{class_id}')
                tc    = train_counts.get(class_id, 0)
                vc    = val_counts.get(class_id, 0)
                xc    = test_counts.get(class_id, 0)
                total = tc + vc + xc
                f.write(f"{name:<15} {tc:>14,} {vc:>14,} {xc:>14,} {total:>14,}\n")

        print(f"Analysis complete! Results saved to {self.output_dir}")

    # ------------------------------------------------------------------
    # Copy files into HFX3D_Splits/
    # ------------------------------------------------------------------

    def create_splits_folder(self, splits_root=None):
        """
        Copy LAS files into HFX3D_Splits/{train,val,test}/ sub-folders.
        Default location: data/HFX3D_Splits/ (sibling of this script).
        """
        if splits_root is None:
            splits_root = Path(__file__).parent / "HFX3D_Splits"
        else:
            splits_root = Path(splits_root)

        if splits_root.exists():
            print(f"HFX3D_Splits already exists at {splits_root.resolve()}, skipping copy.")
            return splits_root

        for sub in ('train', 'val', 'test'):
            (splits_root / sub).mkdir(parents=True, exist_ok=True)

        split_map = {
            'train': self.train_buildings,
            'val':   self.val_buildings,
            'test':  self.test_buildings,
        }

        las_files = list(self.data_folder.glob("*.las"))
        copied = {'train': [], 'val': [], 'test': []}
        skipped = []

        for las_path in las_files:
            filename = las_path.name
            dest_split = None
            for split, buildings in split_map.items():
                if self._match_building(filename, buildings):
                    dest_split = split
                    break

            if dest_split is None:
                skipped.append(filename)
                continue

            dest = splits_root / dest_split / filename
            if dest.exists():
                print(f"  [skip — exists] {dest_split}/{filename}")
            else:
                print(f"  Copying → {dest_split}/{filename}")
                shutil.copy2(las_path, dest)
            copied[dest_split].append(filename)

        print("\nHFX3D_Splits summary:")
        for split, files in copied.items():
            print(f"  {split:<6}: {len(files)} files")
        if skipped:
            print(f"  Skipped (no split match): {skipped}")
        print(f"  Folder: {splits_root.resolve()}")
        return splits_root


def main():
    parser = argparse.ArgumentParser(
        description='Analyse and visualise the HFX3D train/val/test split')
    parser.add_argument('las_dir', help='Directory containing .las files')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for plots (default: split_analysis_output/)')
    parser.add_argument('--no-copy', action='store_true',
                        help='Skip copying files into HFX3D_Splits/')
    args = parser.parse_args()

    analyzer = TrainTestSplitAnalyzer(args.las_dir, output_dir=args.output_dir)
    results = analyzer.analyze_dataset()
    if results:
        analyzer.visualize_split(results)
    if not args.no_copy:
        analyzer.create_splits_folder()


if __name__ == "__main__":
    main()
