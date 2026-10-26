# MTP-2: Low-Cost Automated Tomato Sorting System

Master's Thesis Project (Stage 2) — IIT Bombay. Extends the Stage-1 classification
system into a full detection pipeline trained and validated on real chamber footage,
and deployed on a Raspberry Pi 4 for edge inference.

Full write-up: see `Report/main.pdf` (or `Report/main.tex` for the LaTeX source).

## What's in this repo

This repo tracks code only. Captured images, trained model weights, and generated
datasets are excluded via `.gitignore` (they're large and reproducible from the
scripts below) — see **Reproducing the pipeline** to regenerate them.

```
.
├── capture_script.py, capture_test_script.py   # chamber image capture (training / test sessions)
├── YOLO_SAM/
│   ├── annotate.py                              # SAM2 full-frame auto-annotation (colour-filtered)
│   ├── split_dataset.py                         # train/val split BY TOMATO (avoids frame-level leakage)
│   ├── training_all3_atonce.py                  # 3-class YOLO11n fine-tuning
│   ├── testing_models.py, overripe_test.py      # held-out / cross-session evaluation
│   └── visualise_sam_results.py                 # draws SAM2 boxes back onto frames for QA
├── live_inference.py                            # live webcam inference (PC)
├── demo_capture.py                               # manual press-to-capture demo (6-view fusion)
├── triggered_pi.py                               # auto presence/motion-triggered inference (Pi)
├── bench.py, thread_bench_raw.py, tflite_bench.py  # latency benchmarking (NCNN threads, INT8 vs FP32)
├── export_int8.py                                # TFLite INT8 export (evaluated, not used in final model)
└── Report/                                       # LaTeX source + compiled PDF
```

## Pipeline overview

1. **Capture** — `capture_script.py` captures 6 rotational views per tomato under a
   fixed white-light chamber preset (direct + mirror view in one frame).
2. **Annotate** — `YOLO_SAM/annotate.py` runs SAM2 in automatic mode on each full
   frame, filters candidate masks by colour (hue/saturation appropriate to the
   class) to reject background/mirror-frame false positives, and keeps the top-2
   masks per frame as the direct and mirror tomato instances.
3. **Split** — `YOLO_SAM/split_dataset.py` splits train/val **by tomato**, not by
   frame, so all 6 views of one fruit stay on the same side of the split.
4. **Train** — `YOLO_SAM/training_all3_atonce.py` fine-tunes YOLO11n (COCO-initialized)
   on the 3-class chamber dataset (unripe / ripe / overripe).
5. **Evaluate** — `YOLO_SAM/testing_models.py` checks held-out, cross-session
   accuracy (a separate capture session from training) rather than trusting
   same-session validation alone.
6. **Deploy** — model exported to NCNN (FP32), benchmarked and thread-tuned on a
   Raspberry Pi 4, run via `demo_capture.py` (manual) or `triggered_pi.py`
   (automatic presence-triggered, no continuous inference).

## Key results (see Report for full detail)

- Diagnosed the Stage-1 model's real-chamber failure as a colour/lighting
  mismatch from black-background training data — not model size or background
  alone (see Report Chapter 4).
- Final model: YOLO11n, held out cross-session — ripe 3/3 tomatoes, unripe 6/6
  tomatoes correct.
- Raspberry Pi 4 (NCNN, FP32, 4 threads): 362 ms/inference, predictions verified
  identical to the PC model on all held-out test frames. INT8 quantization was
  tested and rejected (slower and less accurate on the ripe/overripe boundary).

## Reproducing the pipeline

```bash
pip install -r requirements.txt

# 1. Capture chamber images (repeat per class: unripe / ripe / overripe)
python capture_script.py

# 2. Annotate with SAM2 (requires the SAM2 checkpoint — see YOLO_SAM/annotate.py for the path)
python YOLO_SAM/annotate.py

# 3. Split by tomato, then train
python YOLO_SAM/split_dataset.py
python YOLO_SAM/training_all3_atonce.py

# 4. Evaluate on held-out data
python YOLO_SAM/testing_models.py
```

For Raspberry Pi deployment, export the trained model to NCNN
(`model.export(format="ncnn")` in Ultralytics), copy the exported folder and
`triggered_pi.py` to the Pi, and run `python triggered_pi.py` with a venv that
has `ultralytics` and `ncnn` installed.

## Hardware

- Chamber: single USB webcam + mirror (captures direct and reflected view in
  one frame), white LED strip lighting (ESP32-controlled, own `ChamberLED`
  Wi-Fi network — see Report Chapter 8 for lighting calibration).
- Deployment: Raspberry Pi 4 Model B (8 GB RAM), 64-bit Raspberry Pi OS.

## Status / open items

- Overripe cross-session generalization testing — results pending.
- Sustained-load / thermal testing on the Pi — not yet run.
- reCamera (dedicated AI-accelerator camera) — evaluated on paper, not ordered;
  pending a defined throughput requirement.
