# MealAssist: AI Nutrition Assistant

Upload a photo of your meal and get the detected food, real nutrition info, and short
AI-generated feedback, no manual searching or entering food items required.

Built for the CM3070 Final Year Project (University of London), using the "Orchestrating AI
Models to Achieve a Goal" template: several pre-trained AI models are chained together
instead of training one large model.

## How it works

1. **Detection**: YOLO11s, fine-tuned on [UEC FOOD 256](http://foodcam.mobi/dataset256.html)
   (256 food classes), detects the food in the photo.
2. **Captioning**: BLIP describes the whole image, and also helps correct low-confidence
   detections.
3. **Nutrition lookup**: [API Ninjas](https://api-ninjas.com/api/nutrition) returns real
   nutrition values; if it doesn't recognise a specific dish name, the app retries using the
   BLIP caption instead.
4. **Feedback**: a local Ollama LLM (Llama 3) explains the values in plain language. Its
   output is checked by a rule-based filter that catches and replaces any made-up numbers.

Photos are processed in memory only and are never saved to disk.

## Real results

- **Detection accuracy** (held-out validation set): mAP50 = 0.561, mAP50-95 = 0.430
- **29/29 automated tests passing** (unit + integration)

Training charts and confusion matrix are in [`docs/training_results/`](docs/training_results/).

## Setup

```bash
git clone <this-repo-url>
cd MealAssist
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your nutrition API key (get a free one at [api-ninjas.com](https://api-ninjas.com)):

```bash
export API_NINJAS_KEY="your_key_here"   # Windows: $env:API_NINJAS_KEY="your_key_here"
```

Install [Ollama](https://ollama.com) and run:

```bash
ollama pull llama3
ollama serve
```

`models/best.pt` isn't included here (too large for git). Either train your own (see below),
or point `app.py` at `yolo11s.pt` instead for a quick test without food-specific accuracy.

## Run it

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Run the tests

```bash
pip install pytest
python -m pytest test_unit.py test_integration.py -v
```

No GPU or real model weights needed, external calls are mocked.

## Train your own detector

Download UEC FOOD 256 from [foodcam.mobi](http://foodcam.mobi/dataset256.html), then:

```bash
python convert_uecfood256_to_yolo.py /path/to/UECFOOD256 /path/to/output --val-ratio 0.15
python check_class_balance.py /path/to/output
python train_food_yolo11.py --data /path/to/output/data.yaml --epochs 20 --batch 8 --imgsz 512
```

Copy the resulting `best.pt` into `models/best.pt`.

## Known limitations

- Accuracy varies across the 256 food classes; some similar-looking dishes get confused.
- No portion-size estimation, values reflect the dish, not the actual quantity.
- Calories and protein aren't shown (premium-only on the free API tier).
- No way yet for a user to correct a wrong detection in the app itself.

Full evaluation and discussion is in the project report.
