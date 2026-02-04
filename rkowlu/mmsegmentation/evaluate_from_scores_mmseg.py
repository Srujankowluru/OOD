import numpy as np
import os
import torch
from glob import glob
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
MODEL_NAME = "DeepLabV3+ (MMSeg)"
# Base directory where all score folders are
BASE_SCORE_DIR = "/fastdata/groupL/datasets/mapillary/validation/mmseg_deeplab_scores_fullres/"
MASK_DIR = "/fastdata/groupL/datasets/mapillary/v1.2/validation/ood_masks/"
OUTPUT_DIR = "./results_mmseg_deeplab_A6000/" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

OOD_METHODS = ["MSP", "MaxLogit", "Energy", "Entropy", "ReAct"]

# --- 2. MAIN EVALUATION LOOP ---
print(f"Starting evaluation for model: {MODEL_NAME} (from pre-computed scores)")

results = {name: {"scores": []} for name in OOD_METHODS}
all_labels = []

# Get a list of all mask files to process
# --- SET TO 10 to match our inference run ---
mask_files = sorted(glob(os.path.join(MASK_DIR, "*.png")))[:10] 

if not mask_files:
    print(f"ERROR: No OOD masks found in {MASK_DIR}")
    exit()

print(f"Processing {len(mask_files)} images...")

for mask_path in tqdm(mask_files):
    try:
        # Load and prepare the ground truth mask
        mask_img = Image.open(mask_path).convert("L")
        mask_gt = (np.array(mask_img) > 127).astype(np.uint8).flatten()
        all_labels.append(mask_gt)

        base_name = os.path.basename(mask_path).replace('.png', '.npy')

        # Load the pre-computed score for EACH method
        for method_name in OOD_METHODS:
            score_path = os.path.join(BASE_SCORE_DIR, f"{method_name}_scores", base_name)
            ood_score = np.load(score_path)
            results[method_name]["scores"].append(ood_score.flatten())

    except Exception as e:
        print(f"Error on {os.path.basename(mask_path)}: {e}")

# --- 3. CALCULATE METRICS & PLOT (All on CPU) ---
print("Concatenating data...")
all_labels = np.concatenate(all_labels)
for name in results:
    results[name]["scores"] = np.concatenate(results[name]["scores"])

print("Calculating metrics and generating plots...")

plt.figure(figsize=(10, 8))
plt.figure(1); plt.plot([0, 1], [0, 1], 'k--', lw=2); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'OOD Detection ROC - {MODEL_NAME}')
plt.figure(2); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'OOD Detection PR - {MODEL_NAME}')
summary_table = []

for method_name, data in results.items():
    scores = data["scores"]
    auroc = roc_auc_score(all_labels, scores)
    ap = average_precision_score(all_labels, scores)
    fpr, tpr, _ = roc_curve(all_labels, scores)
    fpr95 = fpr[np.where(tpr >= 0.95)[0][0]] if np.any(tpr >= 0.95) else 1.0
    summary_table.append([method_name, auroc*100, ap*100, fpr95*100])
    plt.figure(1); plt.plot(fpr, tpr, lw=2, label=f'{method_name} (AUROC={auroc*100:.1f}%)')
    plt.figure(2); precision, recall, _ = precision_recall_curve(all_labels, scores); plt.plot(recall, precision, lw=2, label=f'{method_name} (AP={ap*100:.1f}%)')
    plt.figure(3); plt.clf()
    scores_id = scores[all_labels == 0]; scores_ood = scores[all_labels == 1]
    n_sample = min(50000, len(scores_id), len(scores_ood))
    if n_sample > 0:
         id_sample = np.random.choice(scores_id, n_sample, replace=False); ood_sample = np.random.choice(scores_ood, n_sample, replace=False)
         sns.histplot(id_sample, color='blue', label='ID', kde=True, stat='density', common_norm=False)
         sns.histplot(ood_sample, color='red', label='OOD', kde=True, stat='density', common_norm=False)
         plt.title(f'Score Distribution - {MODEL_NAME} + {method_name}'); plt.legend(); plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{method_name}.png"))
    else:
        print(f"Skipping histogram for {method_name}, not enough data.")

plt.figure(1); plt.legend(loc="lower right"); plt.savefig(os.path.join(OUTPUT_DIR, f"combined_roc_{MODEL_NAME}.png"))
plt.figure(2); plt.legend(loc="upper right"); plt.savefig(os.path.join(OUTPUT_DIR, f"combined_pr_{MODEL_NAME}.png"))

# --- 4. PRINT FINAL SUMMARY TABLE ---
print(f"\n=== FINAL RESULTS FOR {MODEL_NAME} ===")
print("+-----------------+------------+------------+------------+")
print(f"| {'Method':<15} | {'AUROC':<10} | {'AP':<10} | {'FPR@95':<10} |")
print("+-----------------+------------+------------+------------+")
for row in summary_table:
    print(f"| {row[0]:<15} | {row[1]:>9.2f}% | {row[2]:>9.2f}% | {row[3]:>9.2f}% |")
print("+-----------------+------------+------------+------------+")
print(f"\nPlots saved to {OUTPUT_DIR}")
