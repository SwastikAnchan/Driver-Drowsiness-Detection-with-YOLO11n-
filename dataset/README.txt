CUSTOM DATASET FOR THE 90-95% QUALITY TARGET

This folder is intentionally empty of training images. The base YOLO11n model is a generic COCO detector; it is not a driver-specific 90-95% system by itself.

Add labeled images and YOLO-format annotation files:
  dataset/images/train/*.jpg
  dataset/labels/train/*.txt
  dataset/images/val/*.jpg
  dataset/labels/val/*.txt

Each label line must be:
  class_id x_center y_center width height
with normalized values 0..1.

Classes in this example:
  0 = person
  1 = cell_phone

For the best project-specific result, include:
- normal driving
- phone in hand
- phone near face
- phone partly hidden
- passenger holding a phone
- no phone
- different camera positions
- daytime/nighttime
- glare, blur, sunglasses, masks, partial occlusion
- multiple drivers/passengers

Then train with:
  python train_yolo.py --data dataset/data.yaml --model yolo11n.pt --device 0

After training, evaluate with:
  python evaluate_yolo.py --data dataset/data.yaml --model runs/detect/train/weights/best.pt --device 0 --min-map50 0.90

Do not treat the 90-95%% goal as achieved until the validation and real driving test sets measure it.
