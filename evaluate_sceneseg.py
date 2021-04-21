# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:11:00 2021

@author: mraro
"""

import numpy as np
from sklearn.metrics import average_precision_score


def calc_ap(gt_dict, pr_dict):
    """Average Precision (AP) for scene transitions.
    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.
    Returns:
        AP, mean AP, and a dict of AP for each movie.
    """
    assert gt_dict.keys() == pr_dict.keys()

    AP_dict = dict()
    gt = list()
    pr = list()
    for imdb_id in gt_dict.keys():
        AP_dict[imdb_id] = average_precision_score(gt_dict[imdb_id], pr_dict[imdb_id])
        gt.append(gt_dict[imdb_id])
        pr.append(pr_dict[imdb_id])

    mAP = sum(AP_dict.values()) / len(AP_dict)

    gt = np.concatenate(gt)
    pr = np.concatenate(pr)
    AP = average_precision_score(gt, pr)

    return AP, mAP, AP_dict


def calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict, threshold=0.5):
    """Maximum IoU (Miou) for scene segmentation.
    Miou measures how well the predicted scenes and ground-truth scenes overlap. The descriptions can be found in
    https://arxiv.org/pdf/1510.08893.pdf. Note the length of intersection or union is measured by the number of frames.
    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.
        shot_to_end_frame_dict: End frame index for each shot.
        threshold: A threshold to filter the predictions.
    Returns:
        Mean MIoU, and a dict of MIoU for each movie.
    """
    def iou(x, y):
        s0, e0 = x
        s1, e1 = y
        smin, smax = (s0, s1) if s1 > s0 else (s1, s0)
        emin, emax = (e0, e1) if e1 > e0 else (e1, e0)
        return (emin - smax + 1) / (emax - smin + 1)

    def scene_frame_ranges(scene_transitions, shot_to_end_frame):
        end_shots = np.where(scene_transitions)[0]
        scenes = np.zeros((len(end_shots) + 1, 2), dtype=end_shots.dtype)
        scenes[:-1, 1] = shot_to_end_frame[end_shots]
        scenes[-1, 1] = shot_to_end_frame[len(scene_transitions)]
        scenes[1:, 0] = scenes[:-1, 1] + 1
        return scenes

    def miou(gt_array, pr_array, shot_to_end_frame):
        gt_scenes = scene_frame_ranges(gt_array, shot_to_end_frame)
        pr_scenes = scene_frame_ranges(pr_array >= threshold, shot_to_end_frame)
        assert gt_scenes[-1, -1] == pr_scenes[-1, -1]

        m = gt_scenes.shape[0]
        n = pr_scenes.shape[0]

        # IoU for (gt_scene, pr_scene) pairs
        iou_table = np.zeros((m, n))

        j = 0
        for i in range(m):
            # j start prior to i end
            while pr_scenes[j, 0] <= gt_scenes[i, 1]:
                iou_table[i, j] = iou(gt_scenes[i], pr_scenes[j])
                if j < n - 1:
                    j += 1
                else:
                    break
            # j end prior to (i + 1) start
            if pr_scenes[j, 1] < gt_scenes[i, 1] + 1:
                break
            # j start later than (i + 1) start
            if pr_scenes[j, 0] > gt_scenes[i, 1] + 1:
                j -= 1
        assert np.isnan(iou_table).sum() == 0
        assert iou_table.min() >= 0

        # Miou
        return (iou_table.max(axis=0).mean() + iou_table.max(axis=1).mean()) / 2

    assert gt_dict.keys() == pr_dict.keys()

    miou_dict = dict()

    for imdb_id in gt_dict.keys():
        miou_dict[imdb_id] = miou(gt_dict[imdb_id], pr_dict[imdb_id], shot_to_end_frame_dict[imdb_id])
    mean_miou = sum(miou_dict.values()) / len(miou_dict)

    return mean_miou, miou_dict


def calc_precision_recall(gt_dict, pr_dict, threshold=0.5):
    """Precision, Recall and F1 for scene transitions at a given threshold.
    Args:
        gt_dict: Scene transition ground-truths.
        pr_dict: Scene transition predictions.
        threshold: A threshold to filter the predictions.
    Returns:
        Mean Precision, Recall, and F1, per IMDB ID Precisions, Recalls, and F1 scores.
    """
    def precision_recall(gt_array, pr_array):
        tp_fn = gt_array == 1
        tp_fp = pr_array >= threshold

        tps = (tp_fn & tp_fp).sum()

        precision = tps / tp_fp.sum()
        recall = tps / tp_fn.sum()

        return np.nan_to_num(precision), np.nan_to_num(recall)

    assert gt_dict.keys() == pr_dict.keys()

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()

    for imdb_id in gt_dict.keys():
        p, r = precision_recall(gt_dict[imdb_id], pr_dict[imdb_id])
        precision_dict[imdb_id] = p
        recall_dict[imdb_id] = r
        fscore_dict[imdb_id] = 2 * p * r / (p + r)

    n = len(gt_dict)
    mean_precision = sum(precision_dict.values()) / n
    mean_recall = sum(recall_dict.values()) / n
    mean_fscore = sum(fscore_dict.values()) / n

    return mean_precision, mean_recall, mean_fscore, precision_dict, recall_dict, fscore_dict


if __name__ == "__main__":
    import os
    import sys
    import glob
    import json
    import pickle

    data_dir = sys.argv[1]
    filenames = glob.glob(os.path.join(data_dir, "tt*.pkl"))

    print("# of IMDB IDs:", len(filenames))

    gt_dict = dict()
    pr_dict = dict()
    shot_to_end_frame_dict = dict()

    for fn in filenames:
        x = pickle.load(open(fn, "rb"))

        gt_dict[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"]
        pr_dict[x["imdb_id"]] = x["scene_transition_boundary_prediction"]
        shot_to_end_frame_dict[x["imdb_id"]] = x["shot_end_frame"]

    scores = dict()

    scores["AP"], scores["mAP"], _ = calc_ap(gt_dict, pr_dict)
    scores["Miou"], _ = calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict)
    scores["Precision"], scores["Recall"], scores["F1"], *_ = calc_precision_recall(gt_dict, pr_dict)

    print("Scores:", json.dumps(scores, indent=4))