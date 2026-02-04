#!/bin/bash
# Final run: all OOD methods on Mapillary validation (originally planned data) with SegFormer.
# Same data as DeepLab final run; output -> final_run_segformer/.

cd "$(dirname "$0")"

SEGFORMER_ROOT="$(pwd)"
DEEPLAB_ROOT="$(dirname "$SEGFORMER_ROOT")/DeepLabV3Plus-Pytorch"
IMAGE_LIST="${DEEPLAB_ROOT}/val_list.txt"
MASK_DIR="/fastdata/groupL/datasets/mapillary/v1.2/validation/ood_masks/"
OUTPUT_DIR="final_run_segformer"
GALLERY_DIR="./galleries"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

echo "Final SegFormer run: Mapillary data -> $OUTPUT_DIR"
echo "Image list: $IMAGE_LIST"
echo "Gallery dir: $GALLERY_DIR"
echo ""

"$PYTHON" run_all_methods_segformer.py \
  --image-list "$IMAGE_LIST" \
  --mask-dir "$MASK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --gallery-dir "$GALLERY_DIR" \
  --low-mem \
  --plot \
  --save-vis 20

echo ""
echo "Done. Results in $OUTPUT_DIR/"
