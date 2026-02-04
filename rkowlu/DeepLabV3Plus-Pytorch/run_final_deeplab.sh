#!/bin/bash
# Final run: all OOD methods on Mapillary validation (originally planned data).
# Uses same model & galleries as sanity check; output -> final_run_deeplab/.

cd "$(dirname "$0")"

MODEL="/visinf/projects_students/groupL/rkowlu/best_deeplabv3plus_resnet101_cityscapes_os16.pth (1).tar"
IMAGE_LIST="val_list.txt"
MASK_DIR="/fastdata/groupL/datasets/mapillary/v1.2/validation/ood_masks/"
OUTPUT_DIR="final_run_deeplab"
GALLERY_DIR="."

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

echo "Final DeepLab run: Mapillary data -> $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Image list: $IMAGE_LIST"
echo ""

"$PYTHON" run_all_methods.py \
  --model "$MODEL" \
  --image-list "$IMAGE_LIST" \
  --mask-dir "$MASK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --gallery-dir "$GALLERY_DIR" \
  --plot \
  --save-vis 20

echo ""
echo "Done. Results in $OUTPUT_DIR/"
