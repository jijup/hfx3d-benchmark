from collections import Counter
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
import laspy
import numpy as np

def load_classification_data(las_path):
    """Load classification data from LAS file."""
    las = laspy.read(las_path)
    return las.classification

def compute_metrics_from_distribution(labels_orig, labels_mod):
    """Compute metrics like accuracy, IoU, and Kappa from class distributions, only for changed classes."""
    # Overall accuracy
    overall_acc = accuracy_score(labels_orig, labels_mod)

    # Cohen's Kappa
    kappa = cohen_kappa_score(labels_orig, labels_mod)

    # Confusion matrix
    cm = confusion_matrix(labels_orig, labels_mod)

    # Class distributions
    dist_orig = Counter(labels_orig)
    dist_mod = Counter(labels_mod)

    # Detect classes with changed counts
    changed_classes = set()
    all_classes = set(dist_orig.keys()).union(dist_mod.keys())
    for cls in all_classes:
        if dist_orig.get(cls, 0) != dist_mod.get(cls, 0):
            changed_classes.add(cls)

    # IoU per changed class only
    all_labels = np.unique(np.concatenate((labels_orig, labels_mod)))
    iou_per_class = {}
    for i, label in enumerate(all_labels):
        if label in changed_classes:
            tp = cm[i, i]  # True positives
            fp = cm[:, i].sum() - tp  # False positives
            fn = cm[i, :].sum() - tp  # False negatives
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            iou_per_class[label] = round(iou, 4)

    # Mean IoU over changed classes
    valid_ious = [iou for iou in iou_per_class.values()]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    return overall_acc, mean_iou, kappa, iou_per_class, changed_classes

def print_class_distribution(las_path, label=""):
    """Print class distribution for a LAS file."""
    las = laspy.read(las_path)
    counts = Counter(las.classification)

    print(f"\n=== Class Distribution for {label or las_path} ===")
    for class_id in sorted(counts.keys()):
        print(f"  Class {class_id}: {counts[class_id]:,}")

# Main function to load, compute, and print metrics
def compare_labels_using_distributions(original_path, modified_path):
    """Compare class distributions and compute metrics."""
    print("\nLoading class distributions...")

    # Load classifications from both files
    labels_orig = load_classification_data(original_path)
    labels_mod = load_classification_data(modified_path)

    # Print total point clouds
    print(f"Total point clouds: {len(labels_mod):,}")

    # Compute metrics from class distributions
    print("Computing metrics...")
    overall_acc, mean_iou, kappa, iou_per_class, changed_classes = compute_metrics_from_distribution(labels_orig, labels_mod)

    # Print class distributions
    print_class_distribution(original_path, "Original File")
    print_class_distribution(modified_path, "Modified File")

    # Print metrics
    print("\n===== Class Distribution Comparison Metrics =====")
    print(f"Overall Accuracy: {overall_acc:.4%}")
    print(f"Mean IoU (for changed classes only): {mean_iou:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    if iou_per_class:
        print("\nIoU for Changed Classes:")
        for label, iou in iou_per_class.items():
            print(f"  Class {label:2d}: {iou:.4f}")
    else:
        print("\nNo changes in class distributions — skipping IoU per class.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare classifications between two LAS files')
    parser.add_argument('original', help='Path to original LAS file')
    parser.add_argument('modified', help='Path to modified LAS file')
    args = parser.parse_args()
    compare_labels_using_distributions(args.original, args.modified)