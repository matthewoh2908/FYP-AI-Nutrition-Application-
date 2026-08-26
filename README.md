# MealAssist: AI Nutrition Assistant

MealAssist is an AI Nutrition Assistant that lets a user upload a photo of their meal and
receive an automatically detected food label, real nutrition information, and short,
AI-generated dietary feedback, without manually searching for or entering any food item
themselves.

This project was built for the CM3070 Final Year Project (BSc Computer Science, University
of London), using the **"Orchestrating AI Models to Achieve a Goal"** project template: rather
than training one large model, several pre-trained AI models are connected into a single
pipeline, each handling a different part of the task.

## How it works

1. **Detection** — a YOLO11s model, fine-tuned on the [UEC FOOD 256](http://foodcam.mobi/dataset256.html)
   dataset (256 food categories), detects and labels food items in the uploaded photo.
2. **Captioning** — a BLIP model generates a short natural-language caption of the whole
   image, which is also used to cross-check and correct low-confidence detections.
3. **Nutrition retrieval** — the detected food label is looked up against the
   [API Ninjas](https://api-ninjas.com/api/nutrition) Nutrition API; if a specific dish name
   isn't recognised, the app automatically retries using the BLIP caption instead.
4. **Feedback generation** — a locally-run Ollama LLM (Llama 3) explains the retrieved
   nutrition values in plain language. Its output is checked by a **rule-based filter** that
   scans for any number that wasn't actually in the data it was given, retrying once and then
   falling back to a safe, template-based message if a fabricated number is ever found.
5. **Results interface** — a Flask web app displays the annotated photo, detected food, real
   nutrition values, and the AI-generated feedback.

Uploaded photos are processed entirely in memory and are never written to disk.

## Real results

These are genuine, reproducible results, not estimates.

**Detection accuracy** (YOLO11s fine-tuned on UEC FOOD 256, evaluated on a held-out validation
split of 4,709 images):

| Metric | Result |
|---|---|
| Precision | 0.518 |
| Recall | 0.556 |
| mAP50 | 0.561 |
| mAP50-95 | 0.430 |

Training curves, confusion matrices, and other real training artifacts are in
[`docs/training_results/`](docs/training_results/).

**Automated test suite**: 29/29 tests passing (22 unit tests, 7 integration tests), covering
input validation, nutrition API error handling, the caption cross-check mechanism, and the
hallucination-prevention filter. Run them yourself with the commands below.

## Project structure

```
├── app.py                          # Flask application (main entry point)
├── templates/
│   └── index.html                  # Frontend
├── test_unit.py                    # Unit tests
├── test_integration.py             # Integration tests
├── stub_ml_deps.py                 # Test helper: stubs heavy ML deps so tests run without
│                                    # downloading model weights
├── convert_uecfood256_to_yolo.py   # Converts UEC FOOD 256 into YOLO training format
├── check_class_balance.py          # Reports per-class image counts after conversion
├── train_food_yolo11.py            # Fine-tunes YOLO11s on the converted dataset
├── models/                         # Trained model weights go here (not committed, see below)
├── docs/training_results/          # Real training curves, confusion matrix, etc.
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <this-repo-url>
cd MealAssist
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your nutrition API key

Get a free key from [api-ninjas.com](https://api-ninjas.com), then either copy `.env.example`
to `.env` and fill it in, or set the environment variable directly:

```bash
# Windows PowerShell
$env:API_NINJAS_KEY="your_key_here"

# macOS/Linux
export API_NINJAS_KEY="your_key_here"
```

The app will refuse to start without this set, on purpose, so a missing key fails loudly
rather than silently.

### 3. Install and run Ollama

Download [Ollama](https://ollama.com), then:

```bash
ollama pull llama3
ollama serve
```

### 4. Get the trained detection model

The fine-tuned model weights (`models/best.pt`) are not included in this repository, since
they're too large for a standard git repo. You have two options:

- **Train your own** (see "Training the detector" below), or
- **Use a plain pretrained YOLO model** as a placeholder for testing the rest of the pipeline,
  by pointing `app.py`'s `model = YOLO(...)` line at `yolo11s.pt` instead of `models/best.pt`
  (Ultralytics will download it automatically). Note this will not have the food-specific
  fine-tuning, so detection will be far less accurate.

### 5. Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Running the tests

```bash
pip install pytest
python -m pytest test_unit.py test_integration.py -v
```

All 29 tests should pass. The test suite mocks external calls (the nutrition API, Ollama) and
stubs the ML model imports, so it runs in under a second and doesn't need real model weights,
a GPU, or network access.

## Training the detector

The UEC FOOD 256 dataset is not included in this repository (it's several GB). Download it
from [foodcam.mobi/dataset256.html](http://foodcam.mobi/dataset256.html), then:

```bash
# 1. Convert the dataset into YOLO training format
python convert_uecfood256_to_yolo.py /path/to/UECFOOD256 /path/to/output --val-ratio 0.15

# 2. Check class balance before training (flags any class with too few images)
python check_class_balance.py /path/to/output

# 3. Fine-tune YOLO11s
python train_food_yolo11.py --data /path/to/output/data.yaml --epochs 20 --batch 8 --imgsz 512
```

The `--batch`, `--imgsz`, and `--epochs` values above reflect real constraints from training on
an 8GB GPU (see the project report for the full training story, including the out-of-memory
errors this configuration was chosen to avoid). Adjust these upward if you have more VRAM
available.

Once training finishes, copy the resulting `best.pt` (found under
`runs/food_yolo11/<run-name>/weights/best.pt`) into `models/best.pt`.

## Known limitations

- Detection accuracy varies considerably across the 256 food classes; some visually similar
  dishes (e.g. several cutlet variants) are frequently confused with each other.
- No portion-size estimation is implemented; nutrition values reflect the detected dish, not
  the actual quantity in a photograph.
- Calories and protein are not shown, since these fields are premium-only on the free tier of
  the nutrition API used.
- There is currently no in-app way for a user to correct a wrong detection before nutrition
  values are calculated.

A full, evidence-based evaluation of these limitations, including real accuracy testing,
usability testing with real users, and a critical discussion of results, is in the project
report.

## Acknowledgements

- Kawano, Y., & Yanai, K. (2014). Automatic expansion of a food image dataset leveraging
  existing categories with domain adaptation. *ECCV Workshop on Transferring and Adapting
  Source Knowledge in Computer Vision (TASK-CV)*. (UEC FOOD 256 dataset)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Salesforce BLIP](https://huggingface.co/docs/transformers/en/model_doc/blip)
- [API Ninjas Nutrition API](https://api-ninjas.com/api/nutrition)
- [Ollama](https://ollama.com)
