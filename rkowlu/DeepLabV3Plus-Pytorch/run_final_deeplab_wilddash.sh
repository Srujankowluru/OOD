#!/bin/bash
# Final run: all OOD methods on WildDash2 validation (same setup as Mapillary final run).
# Uses same DeepLab model and galleries; generates WildDash image list and OOD masks if missing.

set -e
cd "$(dirname "$0")"

MODEL="/visinf/projects_students/groupL/rkowlu/best_deeplabv3plus_resnet101_cityscapes_os16.pth (1).tar"
WILDDASH_ROOT="/fastdata/groupL/datasets/wilddash"
IMAGE_LIST="wilddash_val_list.txt"
MASK_DIR="${WILDDASH_ROOT}/ood_masks"
OUTPUT_DIR="final_run_deeplab_wilddash"
GALLERY_DIR="."

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

# Generate WildDash image list and OOD masks if not present
if [ ! -f "$IMAGE_LIST" ] || [ ! -d "$MASK_DIR" ] || [ -z "$(ls -A "$MASK_DIR" 2>/dev/null)" ]; then
  echo "Preparing WildDash: image list and OOD masks..."
  "$PYTHON" prepare_wilddash_ood_masks.py \
    --wilddash-root "$WILDDASH_ROOT" \
    --split validation \
    --out-list "$IMAGE_LIST" \
    --out-mask-dir "$MASK_DIR"
  echo ""
fi

if [ ! -f "$IMAGE_LIST" ]; then
  echo "Error: image list not found: $IMAGE_LIST"
  exit 1
fi
if [ ! -d "$MASK_DIR" ]; then
  echo "Error: mask dir not found: $MASK_DIR"
  exit 1
fi

# Same image count as Mapillary final run (300) for comparable evaluation
MAX_IMAGES=300

echo "Final DeepLab run: WildDash validation -> $OUTPUT_DIR (first $MAX_IMAGES images)"
echo "Model: $MODEL"
echo "Image list: $IMAGE_LIST"
echo "Mask dir: $MASK_DIR"
echo ""

"$PYTHON" run_all_methods.py \
  --model "$MODEL" \
  --image-list "$IMAGE_LIST" \
  --mask-dir "$MASK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --gallery-dir "$GALLERY_DIR" \
  --max-images "$MAX_IMAGES" \
  --plot \
  --save-vis 20

echo ""
echo "Done. Results in $OUTPUT_DIR/"
