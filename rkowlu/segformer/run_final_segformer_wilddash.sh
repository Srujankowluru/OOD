#!/bin/bash
# Final run: all OOD methods on WildDash2 validation with SegFormer.
# Same data as DeepLab WildDash run (image list + OOD masks); output -> final_run_segformer_wilddash/.

cd "$(dirname "$0")"

SEGFORMER_ROOT="$(pwd)"
DEEPLAB_ROOT="$(dirname "$SEGFORMER_ROOT")/DeepLabV3Plus-Pytorch"
IMAGE_LIST="${DEEPLAB_ROOT}/wilddash_val_list.txt"
MASK_DIR="/fastdata/groupL/datasets/wilddash/ood_masks"
OUTPUT_DIR="final_run_segformer_wilddash"
GALLERY_DIR="./galleries"
MAX_IMAGES=300

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

if [ ! -f "$IMAGE_LIST" ]; then
  echo "Error: WildDash image list not found: $IMAGE_LIST"
  echo "Run DeepLab prepare first: cd $DEEPLAB_ROOT && python prepare_wilddash_ood_masks.py ..."
  exit 1
fi
if [ ! -d "$MASK_DIR" ]; then
  echo "Error: WildDash OOD masks not found: $MASK_DIR"
  exit 1
fi

echo "Final SegFormer run: WildDash validation -> $OUTPUT_DIR (first $MAX_IMAGES images)"
echo "Image list: $IMAGE_LIST"
echo "Mask dir: $MASK_DIR"
echo "Gallery dir: $GALLERY_DIR"
echo ""

"$PYTHON" run_all_methods_segformer.py \
  --image-list "$IMAGE_LIST" \
  --mask-dir "$MASK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --gallery-dir "$GALLERY_DIR" \
  --max-images "$MAX_IMAGES" \
  --low-mem \
  --plot \
  --save-vis 20

echo ""
echo "Done. Results in $OUTPUT_DIR/"
