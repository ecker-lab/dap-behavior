from collections import defaultdict
import json
import multiprocessing
import time

import numpy as np
from dap_behavior.eval.mmaction import metrics
from dap_behavior.eval.mmaction.ava_utils import read_exclusions, print_time, tpfp_single

def load_ava_results(results_file):
    """"
    Load logits from disk. If there are separate indices and scores files, the argument should be the indices file.
    """

    if results_file.endswith("npz"):
        with(np.load(results_file)) as data:
            scores = data["scores"]
            scores2idxs = data["indices"]
    else:
        scores2idxs = np.memmap(results_file, dtype=np.int32, mode="r")
        scores = np.memmap(results_file.replace("indices", "scores"), dtype=np.float32, mode="r")
        scores = scores.reshape(scores2idxs.shape[0], -1)

    print("scores", scores.shape, "scores2idxs", scores2idxs.shape)

    if len(scores.shape) > 2:
        scores = np.mean(scores, axis=1) # (samples, classes)

    return dict(scores_avg=scores, scores2idxs=scores2idxs)

def ava_map(scores_avg, scores2idxs, annot_join, gt, label_file, exclude_file, agg_metrics={}, verbose=True, ignore_empty_frames=False, info={}, parallelize="none"):
    """"

    Calculate mAP for AVA dataset.
    
    Args:
    scores_avg np.array(samples, classes): raw logits for each frame
    scores2idxs np.array(samples): indices of the bounding boxes corresponding to the logits
    annot_join pd.DataFrame: bounding boxes for each prediction
    gt pd.DataFrame: ground truth bounding boxes with labels in AVA format
    label_file str: path to the json label file. Format: list of dicts with keys 'id' (starting with 0), 'name' and 'label_type' (used for calucalting agg_metrics).
    exclude_file str: path to the exclusion file
    agg_metrics dict[str, list[str]]: which labels to aggregate to which categories
    verbose bool: whether to print detailed information
    ignore_empty_frames bool: whether to ignore empty frames
    info dict: additional information to include in the result dict
    
    Returns:
    dict: mAP results for each class and overall mAP
    """

    pred_boxes = defaultdict(list)
    pred_scores = defaultdict(list)
    pred_labels = defaultdict(list)

    with open(label_file) as f:
        categories = json.load(f)

    if annot_join is None:
        # This assumes that the scores2idxs are the same as the gt indices
        annot_join = gt[["filename", "frame", "entity_id", "x_min", "y_min", "x_max", "y_max"]].drop_duplicates()

    labels = list(sorted(c['id'] for c in categories))

    for idx, scores in zip(scores2idxs, scores_avg):

        try:
            row = annot_join.loc[idx]
        except KeyError:
            continue

        entry = f"test/images/{row['filename']},{row['frame']:04d}"

        for i, score in enumerate(scores):
            pred_boxes[entry].append(row[['x_min', 'y_min', 'x_max', 'y_max']].to_list())
            pred_scores[entry].append(score)
            pred_labels[entry].append(labels[i])

    gt_bboxes = defaultdict(list)
    gt_labels = defaultdict(list)

    for idx, row in gt.iterrows():
        entry = f"test/images/{row['filename']},{row['frame']:04d}"
        gt_bboxes[entry].append(row[['x_min', 'y_min', 'x_max', 'y_max']].to_list())
        gt_labels[entry].append(row['label'])

    boxes = pred_boxes
    labels = pred_labels
    scores = pred_scores

    agg_metrics = defaultdict(list)
    for cat in categories:
        if 'label_type' in cat:
            agg_metrics[cat['label_type']].append(cat['name'])

    class_whitelist = set(cat['id'] for cat in categories)

    with open(exclude_file) as f:
        excluded_keys = read_exclusions(f)

    start = time.time()
    all_gt_labels = np.concatenate(list(gt_labels.values()))
    gt_count = {k: np.sum(all_gt_labels == k) for k in class_whitelist}

    if verbose:
        print(gt_count)

    if ignore_empty_frames:
        tups = [(gt_bboxes[k], gt_labels[k], boxes[k], labels[k], scores[k])
                for k in gt_bboxes if k not in excluded_keys]
    else:
        tups = [(gt_bboxes.get(k, np.zeros((0, 4), dtype=np.float32)),
                    gt_labels.get(k, []), boxes[k], labels[k], scores[k])
                for k in boxes if k not in excluded_keys]
    if parallelize == "none":
        rets = [tpfp_single(tup) for tup in tups]
    elif parallelize == "multiprocessing":
        pool = multiprocessing.Pool(32)
        rets = pool.map(tpfp_single, tups)
    else:
        raise ValueError(f"Unknown parallelize method: {parallelize}")

    if verbose:
        print_time('Calculating TP/FP', start)

    start = time.time()
    scores, tpfps = defaultdict(list), defaultdict(list)
    for score, tpfp in rets:
        for k in score:
            scores[k].append(score[k])
            tpfps[k].append(tpfp[k])

    cls_AP = []
    for k in scores:
        scores[k] = np.concatenate(scores[k])
        tpfps[k] = np.concatenate(tpfps[k])
        precision, recall = metrics.compute_precision_recall(
            scores[k], tpfps[k], gt_count[k])
        ap = metrics.compute_average_precision(precision, recall)
        class_name = [x['name'] for x in categories if x['id'] == k]
        assert len(class_name) == 1
        class_name = class_name[0]
        cls_AP.append((k, class_name, ap))
    if verbose:
        print_time('Run Evaluator', start)


        print('Per-class results: ', flush=True)
        for k, class_name, ap in cls_AP:
            print(f'Index: {k}, Action: {class_name}: AP: {ap:.4f};', flush=True)

    overall = np.nanmean([x[2] for x in cls_AP])

    if verbose:
        print('Overall Results: ', flush=True)
        print(f'Overall mAP: {overall:.4f}', flush=True)

    result_dict = {x[1]: x[2] for x in cls_AP}

    result = {**info}
    result["_overall"] = overall
    for agg_name, labels in agg_metrics.items():
        result[agg_name] = np.nanmean([result_dict[label] for label in labels])

    result.update(result_dict)

    return result