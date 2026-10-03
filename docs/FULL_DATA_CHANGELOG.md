# Full-data configuration record — 2026-10-03

- VOC: 10,582 train_aug training images; 1,449 official validation images;
  10,582 training images for every CAM/seed evaluation stage.
- COCO 2014: 82,783 training images; 40,504 validation images; 82,783 training
  images for every CAM/seed evaluation stage. ID lists are generated from the
  official annotation JSONs with all images retained.
- D0_no_dino, B0_dinov2_control, B1_partial_labels, A1_anchored_graph and
  AB1_combined share complete lists and the same schedule within each dataset.
- Batch size is 4. VOC uses 24,000 updates; COCO uses 187,753 updates. Warmup
  and target-activation steps are specified in each dataset's JSON configs.
- Exact official membership, uniqueness and train/validation separation are
  verified before training. The final incomplete batch is retained. Completion
  requires observed coverage of every training image.
- Training reads RGB images and image-level labels. Dense masks are used for
  evaluation and visualization. COCO zero-tag images are retained and receive
  background targets.
- Model channels, confusion matrices and CRF output dimensions follow the
  dataset: 21 classes for VOC and 81 for COCO, including background.
- COCO masks retain the original train2014/val2014 image stem. Mask values must
  be background 0, contiguous foreground indices 1..80, or ignored value 255.
- COCO text attributes require a matching 80-class bank or descriptor JSON.
  Their shapes are verified; the VOC attribute bank is used only for VOC.
- tools/configure.py updates VOC and COCO resource paths independently.
  tools/check_data.py checks full image/mask file coverage and image-level tags.
- Per-image confusion matrices use disk-backed int32 buffers with range checks;
  global sums use int64. Evaluation validates each checkpoint's saved config.
- Fixed visualization examples are used only for display. Explicit diagnostics
  report no benchmark mIoU and do not replace complete evaluation.
- The package includes method diagrams, source, configs and full VOC ID lists.
  Full-data training and evaluation results are pending.

PACKAGE_QA.json records CPU checks and packaging verification. Actual GPU training
and inference have not been performed. Images, dense masks, pretrained weights,
and COCO text attributes must be supplied before running the relevant commands.
