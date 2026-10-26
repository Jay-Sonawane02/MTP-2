# export_int8.py - export the final model to TFLite INT8 for Pi comparison
# Needs calibration images (a sample of real chamber frames) to set the
# INT8 quantization ranges. Run this on the PC.

from ultralytics import YOLO

MODEL_PATH = r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_coco_v2\weights\best.pt"
DATA_YAML = r"D:\MTP-2\YOLO_SAM\data_3class.yaml"   # used only for calibration images

model = YOLO(MODEL_PATH)

# int8=True triggers TFLite INT8 quantization using the train images listed
# in data_3class.yaml as calibration data.
model.export(format="tflite", imgsz=640, int8=True, data=DATA_YAML)

print("\nDone. Look for a '...best_saved_model' folder next to best.pt, containing")
print("a file like 'best_full_integer_quant.tflite' — that's the INT8 model to")
print("copy over to the Pi.")
