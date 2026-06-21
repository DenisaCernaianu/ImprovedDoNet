import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm
import os
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import morphology as sho 

    
def calculate_metrics_with_overlap(gt_mask, pred_masks_list, iou_threshold=0.7):
  
    gt_ids = [i for i in np.unique(gt_mask) if i > 0]

    if not gt_ids and not pred_masks_list:
        return 1.0, 1.0, 1.0
    if not gt_ids or not pred_masks_list:
        return 0.0, 0.0, 0.0

    total_inter_aji = 0
    total_union_aji = sum((gt_mask == g_id).sum() for g_id in gt_ids)
    
    tp = 0
    matched_pred_indices = set()
    matches = []   

    for g_id in gt_ids:
        gt_inst = (gt_mask == g_id)
        best_iou = 0
        best_idx = -1

        for idx, pred_inst in enumerate(pred_masks_list):
            if idx in matched_pred_indices:
                continue
                
            inter = np.logical_and(gt_inst, pred_inst).sum()
            if inter == 0:
                continue
            union = np.logical_or(gt_inst, pred_inst).sum()
            iou = inter / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx != -1:
            pred_inst = pred_masks_list[best_idx]
            inter_val = np.logical_and(gt_inst, pred_inst).sum()
            
            
            total_inter_aji += inter_val
            total_union_aji += (pred_inst.sum() - inter_val)
            
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_pred_indices.add(best_idx)
                matches.append((g_id, best_idx))   
        else:
            total_union_aji += gt_inst.sum()

    
    for idx, pred_inst in enumerate(pred_masks_list):
        if idx not in matched_pred_indices:
            total_union_aji += pred_inst.sum()

    aji = total_inter_aji / total_union_aji if total_union_aji > 0 else 0.0
    
    prec = tp / len(pred_masks_list) if pred_masks_list else 0.0
    rec = tp / len(gt_ids) if gt_ids else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    dice_list = []

    for g_id, pred_idx in matches:
        g = (gt_mask == g_id)
        p = pred_masks_list[pred_idx]

        inter = np.logical_and(g, p).sum()
        dice_i = 2 * inter / (g.sum() + p.sum() + 1e-6)
        dice_list.append(dice_i)

    mean_dice = np.mean(dice_list) if dice_list else 0.0

    return aji, mean_dice, f1

    
def evaluate_isbi_multi_threshold(gt_json, res_json):
    coco_gt = COCO(gt_json)
    with open(res_json, 'r') as f:
        predictions = json.load(f)

    scores = [p.get('score', 0) for p in predictions]
    print(f"Score range: min={min(scores):.4f}, max={max(scores):.4f}")

    gt_img_ids     = sorted(coco_gt.getImgIds())
    pred_ids_found = sorted(set(p['image_id'] for p in predictions))
    id_map         = {p_id: g_id for p_id, g_id
                      in zip(pred_ids_found, gt_img_ids)}

    filename_to_preds = {0: {}, 1: {}} 
    for p in predictions:
        actual_gt_id = id_map.get(p['image_id'])
        if actual_gt_id is None:
            continue
        fname  = coco_gt.loadImgs(actual_gt_id)[0]['file_name']
        cat_id = p.get('category_id', 0)
        filename_to_preds[cat_id].setdefault(fname, []).append(p)

    thresholds    = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    categories    = {0: "Cell", 1: "Nuclei"}

    results = {
        cat_id: {t: {"aji": [], "dice": [], "f1": []}
                 for t in thresholds}
        for cat_id in categories
    }

    print(f"Evaluating {len(gt_img_ids)} images...")

    for img_id in tqdm(gt_img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        fname    = img_info['file_name']
        h, w     = img_info['height'], img_info['width']

        for cat_id, cat_name in categories.items():
            
            gt_mask = np.zeros((h, w), dtype=np.int32)
            weight_gt=0.3
            anns    = coco_gt.loadAnns(
                coco_gt.getAnnIds(imgIds=img_id, catIds=[cat_id])
            )
            for i, ann in enumerate(anns):
                gt_mask[coco_gt.annToMask(ann) > 0] = i + 1

            raw_preds = filename_to_preds[cat_id].get(fname, [])

            for t in thresholds:
                current_preds = [p for p in raw_preds
                                 if p.get('score', 0) >= t]
                current_preds = sorted(current_preds,
                                       key=lambda x: x.get('score', 0),
                                       reverse=True)

                pred_masks_list = [
                    maskUtils.decode(p['segmentation']).astype(bool)
                    for p in current_preds
                ]

                a, d, f = calculate_metrics_with_overlap(
                    gt_mask, pred_masks_list, iou_threshold=0.7
                )

                results[cat_id][t]["aji"].append(a)
                results[cat_id][t]["dice"].append(d)
                results[cat_id][t]["f1"].append(f)

    for cat_id, cat_name in categories.items():
        print(f"\n{'='*50}")
        print(f"Categoria: {cat_name} (id={cat_id})")
        print(f"{'Thresh':<8} | {'AJI':<8} | {'Dice':<8} | {'F1':<8}")
        print("-" * 50)
        for t in thresholds:
            m_aji  = np.mean(results[cat_id][t]["aji"])
            m_dice = np.mean(results[cat_id][t]["dice"])
            m_f1   = np.mean(results[cat_id][t]["f1"])
            print(f"{t:<8} | {m_aji:.4f} | {m_dice:.4f} | {m_f1:.4f}")
        print("="*50)
        
   
    n_cell_total   = 0
    n_nuclei_total = 0
    for img_id in gt_img_ids:
        n_cell_total   += len(coco_gt.getAnnIds(imgIds=img_id, catIds=[0]))
        n_nuclei_total += len(coco_gt.getAnnIds(imgIds=img_id, catIds=[1]))

    total    = n_cell_total + n_nuclei_total
    w_cell   = n_cell_total   / total if total > 0 else 0.5
    w_nuclei = n_nuclei_total / total if total > 0 else 0.5

  
    print(f"\nPonderi: Cell={w_cell:.3f} ({n_cell_total} instante) | "
          f"Nuclei={w_nuclei:.3f} ({n_nuclei_total} instante)")

    print("\nMedia ponderata dupa nr instante:")
    print(f"{'Thresh':<8} | {'AJI':<8} | {'Dice':<8} | {'F1':<8}")
    print("-" * 45)

    for t in thresholds:
        aji_w  = (np.mean(results[0][t]["aji"])  * w_cell +
              np.mean(results[1][t]["aji"])  * w_nuclei)
        dice_w = (np.mean(results[0][t]["dice"]) * w_cell +
              np.mean(results[1][t]["dice"]) * w_nuclei)
        f1_w   = (np.mean(results[0][t]["f1"])   * w_cell +
              np.mean(results[1][t]["f1"])   * w_nuclei)
        print(f"{t:<8} | {aji_w:.4f} | {dice_w:.4f} | {f1_w:.4f}")

    print("="*45)


    print("\nMedia aritmetica:")
    for t in thresholds:
        aji_a  = (np.mean(results[0][t]["aji"])  + np.mean(results[1][t]["aji"]))  / 2
        dice_a = (np.mean(results[0][t]["dice"]) + np.mean(results[1][t]["dice"])) / 2
        f1_a   = (np.mean(results[0][t]["f1"])   + np.mean(results[1][t]["f1"]))   / 2
        print(f"{t:<8} | {aji_a:.4f} | {dice_a:.4f} | {f1_a:.4f}")   
        



if __name__ == "__main__":
   
    GT = "datasets/isbi/annotations/isbi_test.json"
    RES = "results/trainFinal/results_new/coco_instances_results.json"
    GT_CNSEG="datasets/cnseg/cnseg_test/annotations/cnseg_test.json"
    RES_CNSEG="results/cnseg_eval/results/coco_instances_results.json"
    
    evaluate_isbi_multi_threshold(GT,RES)
