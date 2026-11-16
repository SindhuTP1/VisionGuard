# VisionGuard

**A multi-layer AI hallucination detection prototype with explainable visual validation.**

VisionGuard looks at an image and asks two questions:

1. **Does it look AI-generated?** A trained classifier answers this.
2. **Is anything anatomically wrong?** Extra fingers, extra hands and distorted faces are checked with computer-vision geometry.

It combines the answers into one of three verdicts and shows every piece of evidence on the image: bounding boxes, face and hand meshes, and a Grad-CAM heatmap. You can export the full result as JSON or as a printable PDF report.

> **Research prototype.** Verdicts are signals for a human reviewer, not proof. Every number in this README and in the app comes from a real evaluation run (see [Measured results](#measured-results)). None are typed in by hand.

---

## Contents
- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Using the dashboard](#using-the-dashboard)
- [Measured results](#measured-results)
- [Limitations](#limitations-read-this)
- [Retraining and re-evaluating](#retraining-and-re-evaluating)
- [Project structure](#project-structure)
- [API](#api)
- [Data and licences](#data-and-licences)

---

## What it does

| Verdict | Meaning |
|---|---|
| 🟢 **VERIFIED SECURE** | No anomaly found, and the classifier is at least 65% sure the image is real |
| 🟡 **HUMAN REVIEW** | The evidence is uncertain (below 65%) or the layers disagree |
| 🔴 **ANOMALY CONFIRMED** | A structural problem or a confident AI-generated prediction, at 65% or above |

Anomaly categories: `normal`, `anatomical_extra`, `floating_anomaly`, `melting_geometry`, `text_distortion`.

## How it works

```
Image upload
  → preprocessing (EXIF rotation, size limits, decompression-bomb guard)
  → Layer 1  YOLOv8n-pose: people + body keypoints  ·  YOLOv8n (COCO): objects
  → Layer 2  MediaPipe: face mesh (478 pts) + symmetry  ·  hands (21 pts) + finger counting
             + hands-per-person check (hand ↔ pose-wrist matching)
  → Layer 3  MobileNetV3-Small real-vs-AI classifier + Grad-CAM
  → arbitration (findings → class, confidence, verdict)
  → telemetry (probability matrix, entropy, confidence boost) → report
```

| Layer | What it checks | Notes |
|---|---|---|
| **1 · YOLOv8** | Finds people (with pose) and everyday objects | Locates what the other layers inspect. An object detector **can't** tell whether an image is AI-generated. |
| **2 · MediaPipe** | **Face symmetry:** 3D landmark distances with head-tilt (roll) compensation. **Finger count:** the hand is cut out (GrabCut, seeded by the landmarks), then fingertips are counted along its outline and seams between touching fingers are counted inside it. **Hands per person:** each hand is matched to a body-pose wrist. | MediaPipe always fits a 5-finger skeleton, so fingers are counted **independently** of it. When a hand is a fist, a grip or too small, the result is "can't count" instead of a guess. |
| **3 · Classifier** | MobileNetV3-Small (ImageNet weights) fine-tuned on 2,000 images from [AI-vs-Real](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real), plus Grad-CAM | Outputs **real vs AI-generated** only. It looks at five 224px patches of the image at their original resolution. |

**Arbitration rules**
- 0 hands → the finger checks are skipped and the classifier decides.
- A face problem → `melting_geometry`. An extra finger or extra hand → `anatomical_extra`.
- When a geometric finding and the classifier disagree, their confidences are averaged, which usually pushes the case to HUMAN REVIEW.
- Final confidence below 65% → HUMAN REVIEW.

## Quick start

**Requirements:** Python 3.11–3.13, Node.js 20 or later. A CPU is enough; no GPU is needed.

**1. Backend** (first run downloads about 25 MB of model files into `backend/weights/`)
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The classifier checkpoint (`weights/mobilenetv3_small_aivsreal.pt`) is **not** stored in git. Create it once (takes about 5 minutes on CPU):
```bash
python -m training.prepare_dataset      # downloads 2,000 images (~250 MB)
python -m training.train                # trains, evaluates, installs the checkpoint
```

Start the API:
```bash
uvicorn app.main:app --port 8000
```

**2. Frontend** (second terminal)
```bash
cd frontend
npm install
npm run dev
```
Open the address it prints, usually **http://localhost:3000**. If the API runs somewhere other than `localhost:8000`, copy `frontend/.env.example` to `.env.local` and change `NEXT_PUBLIC_API_URL`.

**3. Tests**
```bash
cd backend
.venv\Scripts\python -m pytest tests     # 62 tests
```

## Using the dashboard

**Left: Scan Control Center**
1. **Forensic Detection Engine.** Choose what decides the verdict:
   - *Multi-Layer Arbitration:* all layers.
   - *Biometric Geometry Only:* the classifier is ignored.
   - *Neural Classifier Only:* the geometry checks are ignored.
2. **Detection Sensitivity (0–100%).** 50% uses the calibrated defaults. Higher values multiply geometric evidence by up to ×2 and lower the face-symmetry threshold. The 65% review bar stays fixed.
3. **Upload** a JPG, JPEG, PNG or WEBP image (up to 15 MB), then press **Initialize Target Scan**.
4. **Visualization tabs** switch without re-running the scan:
   - **Original Source**
   - **Layer 1 Bounding Boxes:** objects, faces, people and the analysed region.
   - **Layer 2 Mesh Overlay:** face mesh, hand skeletons with numbered fingertips, and body pose.
   - **Layer 3 Grad-CAM Heatmap:** what pushed the classifier toward its answer, drawn only on the 5 patches it actually examined.
5. **Reset** clears everything.

**Right: System Telemetry**
- **Verdict** with confidence and:
  - **matrix boost:** how far the geometry checks moved the confidence away from the classifier's own score;
  - **entropy:** how uncertain the classifier is (0 bits = certain, 1 bit = coin flip);
  - **cache status:** HIT means this exact image and settings were already analysed;
  - **latency** and the full **rationale**.
- **Layer 1, 2 and 3 cards** with each layer's individual results.
- **Anomaly Category Matrix** (see [Limitations](#limitations-read-this) for where each number comes from).
- **Generate Forensic Archive:**
  - **Export Raw JSON:** the complete API response.
  - **Print Diagnostic PDF:** a formatted report; choose "Save as PDF" in the print dialog.

The header shows a live API latency badge (measured every 5 s). **Model metrics** (`/metrics`) shows the evaluation results below. The original single-page analyzer is still available at `/classic`.

Sample images to try are in `sample_images/`, which isn't in git. File names say what each image really is.

## Measured results

### Classifier (run `20261114-230220`, test set of 320 images never used in training)

| Metric | Value |
|---|---|
| Accuracy | **95.0%** (95% CI 92.5–97.2%) |
| ROC-AUC | 0.987 |
| F1 (AI-generated = positive) | 0.951 |
| Sent to review at 65% | 6.6% of images |
| **Scale stress test** (all images resized to a 512px short side) | **91.3%** |
| **Shortcut probe** (image size + file size only, no pixels) | **100%** ⚠ |

⚠ **Read the accuracy with care.** In this dataset, AI images are all 1024×1024 and the real ones are mostly about 640×440. Image size alone separates them perfectly. The classifier is protected against this: every image is re-saved at JPEG quality 75 and cropped at its original resolution, with no resizing. The stress test shows it keeps most of its accuracy without the size cue. Accuracy on other datasets is **unknown**, and it does make mistakes on unusual real photos (for example, a TV broadcast frame was flagged as AI at 65%).

### Finger counting on real hands (HaGRID photos, where any "6+ fingers" result is a false alarm)

| Set | Countable hands | Wrongly reported 6+ |
|---|---|---|
| Tuning set (thresholds were set here) | 356 | 0 |
| Held-out set (exposed one bug, then fixed) | 393 | 1 |
| **Fresh test set (final, untouched)** | **513** | **2 (0.39%, 95% CI 0.11–1.4%)** |

Only about 40% of real hands can be counted (38–43% across the three sets). The rest are fists, grips or too small, and get "can't count".

### Anatomy checks on human-labelled AI images (HADM validation set, 4,180 images)

| Check | Caught | False alarms |
|---|---|---|
| Extra hand / arm (per image) | 2 of 146 (1.4%) | 0.12% |
| 6+ fingers (per malformed hand) | 0 of 1,599 | 0.09% |
| Finger-shape (skeleton) check | 24% | 24%, **no better than chance**, so it's **not** used for the verdict |

**In plain words:** the geometry checks rarely raise false alarms, but they also rarely catch real AI anatomy errors. Most AI-image hands are small or holding things, so they can't be counted. A learned artefact detector (`training/train_artifacts.py`, YOLOv8n on HADM labels) is included but was **not trained** for this release, because it needs several hours of CPU time.

Raw results: `backend/runs/<run_id>/metrics.json` and `backend/runs/anatomy-20261115/*.json`.

## Limitations (read this)

- **No model was trained on the 5 anomaly categories.** The matrix shows each value's real source:
  - **Normal:** the classifier's probability that the image is real.
  - **Anatomical Extra / Melting Geometry:** the strongest geometry-check score (a heuristic score, not a calibrated probability).
  - **Floating Anomaly / Text Distortion:** **"not measured"**, because there's no detector for them.
- **The classifier is MobileNetV3-Small, not ResNet.** It was trained on a single small dataset, so expect errors on screenshots, video frames, artwork and heavily edited photos.
- **Arm-proportion checks are shown but not used.** Foreshortening (an arm pointing at the camera) makes them unreliable.
- **Face symmetry assumes a neutral face that isn't turned away.** Strongly turned heads are skipped, and expressions such as winks or smirks can raise the score.
- **CPU timing:** a scan takes about 0.3–3 s; hand analysis is the slowest step.

## Retraining and re-evaluating

All commands run from `backend/` with the venv active.

| Command | What it does |
|---|---|
| `python -m training.prepare_dataset` | Downloads a balanced 1,000 + 1,000 subset of AI-vs-Real, split by storage block so near-duplicates can't leak into the test set |
| `python -m training.train` | AdamW (lr 1e-3, weight decay 1e-4), batch 32, up to 12 epochs, early stopping (patience 3). Writes `runs/<id>/metrics.json` and installs the checkpoint |
| `python -m training.evaluate --run-id <id>` | Re-scores a checkpoint on a split |
| `python -m training.eval_fingers <folder> --workers 6` | Finger-count statistics on a folder of real hand photos |
| `python -m training.eval_hadm --workers 8` | Anatomy checks against HADM labels (needs HADM in `data/hadm/`) |
| `python -m training.prepare_hadm_yolo` then `python -m training.train_artifacts` | Builds the dataset for the learned artefact detector, then trains it (several hours on CPU) |

## Project structure

```
backend/
  app/
    main.py            FastAPI app: /analyze, /health, /metrics, /metrics/anatomy (+ result cache)
    pipeline.py        runs all layers in order; a failing layer is reported, never fatal
    detectors.py       person/pose crop, hand skeleton checks, face symmetry
    anatomy.py         finger counting (outline + seam scanline)
    body.py            hands-per-person (pose wrists)
    objects.py         YOLOv8 COCO object detection
    classifier.py      MobileNetV3 inference + Grad-CAM
    artifacts.py       optional learned artefact detector
    arbitration.py     findings → verdict, probability matrix, entropy, boost
    config.py          every threshold and setting (override with VG_* env vars)
    model_loader.py    lazy, thread-safe model loading; missing models are reported, never faked
    schemas.py         API response models
  training/            dataset prep, training, evaluation scripts
  tests/               62 pytest tests
  runs/                metrics from real runs (checkpoints are not in git)
frontend/              Next.js 16 + React 19 + Tailwind 4
  src/components/dashboard/   Dashboard, Visualizer (4 tabs), Telemetry cards, ForensicReport (PDF)
  src/app/             / (dashboard), /metrics, /classic
```

## API

| Endpoint | Description |
|---|---|
| `POST /analyze` | Multipart form fields: `file` (image), `sensitivity` (0–100, default 50), `engine` (`full` \| `geometry` \| `classifier`). Returns the full report. |
| `GET /health` | Status of each model component, plus the decision thresholds |
| `GET /metrics?run_id=` | Classifier evaluation for a training run (`metrics.json`, served unchanged) |
| `GET /metrics/runs` | List of training runs |
| `GET /metrics/anatomy` | Latest anatomy evaluation files |

Interactive API docs: **http://localhost:8000/docs**

## Data and licences

| Data | Used for | Notes |
|---|---|---|
| [Parveshiiii/AI-vs-Real](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real) (MIT) | Classifier training and testing | Labels: `0` = AI, `1` = real. The card says the dataset is balanced, but it holds 3,333 AI vs 10,666 real images |
| [HaGRID mirror](https://huggingface.co/datasets/Vincent-luo/hagrid-mediapipe-hands) | False-alarm tests on real hands | Kept locally only |
| [HADM](https://github.com/wangkaihong/HADM) | Anatomy evaluation (and optional detector training) | **No licence stated**, so it's kept locally, never redistributed, and nothing trained on it is published |
| YOLOv8 weights ([Ultralytics](https://github.com/ultralytics/ultralytics)) | Layer 1 | AGPL-3.0 |
| MediaPipe Tasks models ([Google](https://ai.google.dev/edge/mediapipe)) | Layer 2 | Apache-2.0 |

Datasets, model weights and checkpoints are excluded from git (see `.gitignore`); they're re-created by the scripts above.
