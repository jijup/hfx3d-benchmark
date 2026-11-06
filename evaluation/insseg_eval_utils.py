"""
Shared ScanNet-v2-style instance segmentation evaluator for HFX3D.

Direct port of Pointcept's InsSegEvaluator/InsSegTester
associate_instances()/evaluate_matches()
(pointcept/engines/hooks/evaluator.py, pointcept/engines/test.py), adapted to
take plain numpy arrays instead of a live trainer/tester object. Same metric,
same overlap thresholds, same min-region-size filter -- every HFX3D
instance-seg result in this project (PointGroup's own test.py run, the
clustering-oracle baselines, the full-scene tiling+merge eval) goes through
this exact same class, so numbers are directly comparable across all of them.

Pure numpy -- no torch/pointcept import -- so this stays importable from
tools/eval_cluster_gt_hfx3d.py's standalone (sklearn-only) environment as
well as from tools/test_fullscene_insseg_hfx3d.py's full Pointcept/CUDA one.
"""

from uuid import uuid4

import numpy as np


class ScanNetInstanceEvaluator:
    def __init__(self, class_names, segment_ignore_index, instance_ignore_index):
        self.class_names = class_names
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.valid_class_names = [
            class_names[i] for i in range(len(class_names)) if i not in segment_ignore_index
        ]
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100

    def associate_instances(self, pred_masks, pred_classes, pred_scores, segment, instance):
        # Sparse (point-index-array) intersections, NOT dense N-length boolean
        # masks. The original port (mirroring Pointcept's InsSegEvaluator/
        # InsSegTester verbatim) stored a full dense mask per predicted
        # instance and recomputed `instance == gt_inst["instance_id"]` (a
        # dense O(N) scan) for every (pred, gt) pair -- fine for Pointcept's
        # normal ~100-250k point crops and this project's clustering-oracle
        # baseline (tens of instances/building), but it OOM'd running the
        # full-scene PointGroup stitching eval: confirmed via `sstat` showing
        # 67GB RSS against a 64G job limit while scoring a 47.9M-point
        # building with 2309 barely-merged instances (2309 dense masks x
        # ~48MB each is >100GB by itself). Fixed by converting every mask to
        # a sorted point-index array once, and intersecting via
        # np.intersect1d (cost scales with instance size, not building size)
        # -- same matching logic, same output fields (just no "mask" key,
        # which evaluate_matches() never read), so every already-reported
        # number from this evaluator (clustering-oracle DBSCAN/HDBSCAN)
        # stays exactly reproducible.
        # Thing points whose instance is ignore (-1, i.e. removed
        # <200-pt instances) are void too, as in the official ScanNet evaluator.
        void_mask = np.isin(segment, self.segment_ignore_index) | (instance == self.instance_ignore_index)

        gt_instances = {name: [] for i, name in enumerate(self.class_names) if i not in self.segment_ignore_index}
        instance_ids, idx, counts = np.unique(instance, return_index=True, return_counts=True)
        segment_ids = segment[idx]

        # Group every point's original index by its GT instance id once
        # (O(N log N)), instead of recomputing `instance == id` per pair.
        order = np.argsort(instance, kind="stable")
        sorted_instance = instance[order]
        starts = np.searchsorted(sorted_instance, instance_ids)
        ends = np.append(starts[1:], len(instance))
        gt_idx_by_id = {}

        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_idx_by_id[instance_ids[i]] = order[starts[i]:ends[i]]
            gt_inst = dict(
                instance_id=instance_ids[i], segment_id=segment_ids[i],
                dist_conf=0.0, med_dist=-1.0, vert_count=counts[i], matched_pred=[],
            )
            gt_instances[self.class_names[segment_ids[i]]].append(gt_inst)

        pred_instances = {name: [] for i, name in enumerate(self.class_names) if i not in self.segment_ignore_index}
        instance_id = 0
        for i in range(len(pred_classes)):
            if pred_classes[i] in self.segment_ignore_index:
                continue
            pred_idx = np.flatnonzero(np.not_equal(pred_masks[i], 0))
            pred_inst = dict(
                uuid=uuid4(), instance_id=instance_id, segment_id=pred_classes[i],
                confidence=pred_scores[i],
            )
            pred_inst["vert_count"] = pred_idx.shape[0]
            pred_inst["void_intersection"] = int(np.count_nonzero(void_mask[pred_idx]))
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue
            segment_name = self.class_names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_inst in gt_instances[segment_name]:
                gt_idx = gt_idx_by_id[gt_inst["instance_id"]]
                intersection = np.intersect1d(pred_idx, gt_idx, assume_unique=True).shape[0]
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_size = self.min_region_sizes
        distance_thresh = float("inf")
        distance_conf = -float("inf")

        ap_table = np.zeros((len(self.valid_class_names), len(overlaps)), float)
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for scene in scenes:
                for label_name in self.valid_class_names:
                    for p in scene["pred"][label_name]:
                        pred_visited[p["uuid"]] = False
            for li, label_name in enumerate(self.valid_class_names):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for scene in scenes:
                    pred_instances = scene["pred"][label_name]
                    gt_instances = [
                        gt for gt in scene["gt"][label_name]
                        if gt["vert_count"] >= min_region_size
                        and gt["med_dist"] <= distance_thresh
                        and gt["dist_conf"] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    for gti, gt in enumerate(gt_instances):
                        found_match = False
                        for pred in gt["matched_pred"]:
                            if pred_visited[pred["uuid"]]:
                                continue
                            overlap = float(pred["intersection"]) / (
                                gt["vert_count"] + pred["vert_count"] - pred["intersection"]
                            )
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match:
                            hard_false_negatives += 1
                    cur_true = cur_true[cur_match]
                    cur_score = cur_score[cur_match]

                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred["matched_gt"]:
                            overlap = float(gt["intersection"]) / (
                                gt["vert_count"] + pred["vert_count"] - gt["intersection"]
                            )
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred["void_intersection"]
                            for gt in pred["matched_gt"]:
                                if gt["segment_id"] in self.segment_ignore_index:
                                    num_ignore += gt["intersection"]
                                if (
                                    gt["vert_count"] < min_region_size
                                    or gt["med_dist"] > distance_thresh
                                    or gt["dist_conf"] < distance_conf
                                ):
                                    num_ignore += gt["intersection"]
                            proportion_ignore = float(num_ignore) / pred["vert_count"]
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(cur_score, pred["confidence"])

                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                if has_gt and has_pred:
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1
                    num_examples = len(y_score_sorted)
                    num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        precision[idx_res] = float(tp) / (tp + fp)
                        recall[idx_res] = float(tp) / (tp + fn)

                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.0)
                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                    ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float("nan")
                ap_table[li, oi] = ap_current

        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict(
            all_ap=np.nanmean(ap_table[:, oAllBut25]),
            all_ap_50=np.nanmean(ap_table[:, o50]),
            all_ap_25=np.nanmean(ap_table[:, o25]),
            classes={},
        )
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = dict(
                ap=np.average(ap_table[li, oAllBut25]),
                ap50=np.average(ap_table[li, o50]),
                ap25=np.average(ap_table[li, o25]),
            )
        return ap_scores
