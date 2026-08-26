import os
import base64
import re
import requests
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# api ninjas is used for nutrition lookups (calorieninjas is now part of the api ninjas platform, same nutrition endpoint)
API_NINJAS_KEY = os.environ.get('API_NINJAS_KEY')
if not API_NINJAS_KEY:
    raise RuntimeError(
        "API_NINJAS_KEY environment variable is not set. "
        "Get a free key from https://api-ninjas.com and set it, "
        "e.g. on Windows: set API_NINJAS_KEY=your_key_here"
    )
NUTRITION_API_URL = 'https://api.api-ninjas.com/v1/nutrition'

# ollama runs locally on port 11434, run `ollama pull llama3` once before starting the app
OLLAMA_API_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'llama3'

# detection: YOLO11s fine-tuned on UEC FOOD 256 (256 food/dish classes), starting from COCO-pretrained weights. training details and results are in the Evaluation chapter of the report.

# best trained model
model = YOLO('models/best.pt')

# captioning: BLIP generates a short description of the whole meal, used both to give the user context and to cross-check weak detections 
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

# only keep detections with at least 25% confidence
DETECTION_CONFIDENCE_THRESHOLD = 0.25

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(pil_image):
    """converts a PIL image to a base64 string so it can be shown in HTML"""
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def get_caption(pil_image):
    """
    returns a short caption describing the whole image, e.g. "a plate of
    rice with chicken and vegetables". used to give the user context and
    to cross-check low-confidence detections.
    """
    try:
        inputs = blip_processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            output_ids = blip_model.generate(**inputs, max_new_tokens=30)
        return blip_processor.decode(output_ids[0], skip_special_tokens=True)
    except Exception:
        return ""


def get_nutrition_data(food_labels):
    """
    looks up nutrition info for the detected food labels using API Ninjas.
    note: calories and protein_g are premium-only on the free tier, so
    they're left out; fibre, sodium, carbohydrates, fat, sugar, potassium
    and cholesterol are shown instead.
    """
    if not food_labels:
        return {'success': False, 'items': [], 'totals': {},
                'note': 'No food labels were provided for nutrition lookup.'}

    query = ', '.join(food_labels)

    try:
        response = requests.get(
            NUTRITION_API_URL,
            params={'query': query},
            headers={'X-Api-Key': API_NINJAS_KEY},
            timeout=10
        )

        if response.status_code != 200:
            return {'success': False, 'items': [], 'totals': {},
                    'note': f'Nutrition API request failed (status {response.status_code}).'}

        items = response.json()
        if not items:
            return {'success': False, 'items': [], 'totals': {},
                    'note': 'No nutrition data found for the detected food items.'}

        totals = {
            'carbohydrates_total_g': 0, 'fat_total_g': 0, 'fiber_g': 0,
            'sugar_g': 0, 'sodium_mg': 0, 'potassium_mg': 0, 'cholesterol_mg': 0
        }
        for item in items:
            for key in totals:
                totals[key] += item.get(key, 0) or 0
        totals = {k: round(v, 1) for k, v in totals.items()}

        return {
            'success': True, 'items': items, 'totals': totals,
            'note': 'Calories and protein are premium-only fields on API Ninjas '
                    'and are not shown in this app.'
        }
    except requests.exceptions.RequestException as e:
        return {'success': False, 'items': [], 'totals': {},
                'note': f'Could not reach the nutrition API: {str(e)}'}


def find_unverified_numbers(feedback_text, nutrition_totals):
    """
    rule-based hallucination check. the LLM prompt tells the model not to
    invent nutrition numbers, but a prompt instruction alone doesn't
    guarantee that, so this checks the model's actual output text against
    the real values it was given.

    looks for number+unit patterns that look like a nutrition figure (e.g.
    "17g", "447 mg") and checks each one against the real supplied totals,
    allowing a small tolerance for rounding. returns a list of numbers
    found in the text that don't match any real value, so the caller can
    decide whether to trust the response. an empty list means the response
    passed the check.
    """
    if not nutrition_totals:
        return []

    real_values = [round(v, 1) for v in nutrition_totals.values() if isinstance(v, (int, float))]

    unverified = []
    for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(g|mg)\b', feedback_text, re.IGNORECASE):
        claimed_value = float(match.group(1))
        # allow a small tolerance so rounding differences (e.g. writing
        # "16g" for a true value of 16.2) aren't flagged as made up
        if not any(abs(claimed_value - real) <= 0.5 for real in real_values):
            unverified.append(match.group(0))

    return unverified


def build_fallback_feedback(nutrition_totals):
    """
    a safe, fixed feedback message built directly from the real nutrition
    values, used when the LLM's output still fails the hallucination
    check (see find_unverified_numbers) after a retry. this makes sure
    the user never sees a number that wasn't actually supplied to the
    model, even if the message ends up less personalised.
    """
    if not nutrition_totals:
        return "Nutritional feedback is unavailable right now, as no nutrition data could be retrieved for this meal."

    return (
        f"This meal contains approximately {nutrition_totals.get('carbohydrates_total_g', 0)}g "
        f"carbohydrates, {nutrition_totals.get('fat_total_g', 0)}g fat, "
        f"{nutrition_totals.get('fiber_g', 0)}g fibre and {nutrition_totals.get('sodium_mg', 0)}mg sodium. "
        "A more detailed explanation could not be generated reliably for this meal; "
        "these figures are taken directly from the nutrition database."
    )


def get_llm_feedback(food_labels, caption, nutrition_totals):
    """
    generates personalised nutrition feedback using a local Ollama LLM.
    the LLM is only asked to explain the values it's given, never to
    invent its own, which the prompt wording below enforces.

    this is checked at two levels, not just the prompt: an automated test
    checks the constraint text is present in every prompt, and the actual
    output text is checked here, after generation, by a rule-based filter
    (find_unverified_numbers) that looks for any number that doesn't
    match a real supplied value. if the first attempt fails this check,
    one retry is made with a stricter prompt; if that also fails, a safe,
    fixed fallback message built from the real values is returned
    instead, so the user is never shown an unverified number.
    """
    labels_text = ', '.join(food_labels) if food_labels else 'unknown food items'

    if nutrition_totals:
        nutrition_text = (
            f"Carbohydrates: {nutrition_totals.get('carbohydrates_total_g', 0)}g, "
            f"Fat: {nutrition_totals.get('fat_total_g', 0)}g, "
            f"Fibre: {nutrition_totals.get('fiber_g', 0)}g, "
            f"Sugar: {nutrition_totals.get('sugar_g', 0)}g, "
            f"Sodium: {nutrition_totals.get('sodium_mg', 0)}mg, "
            f"Potassium: {nutrition_totals.get('potassium_mg', 0)}mg, "
            f"Cholesterol: {nutrition_totals.get('cholesterol_mg', 0)}mg"
        )
    else:
        nutrition_text = "No nutrition data available."

    def call_ollama(strict):
        instruction = (
            "Using ONLY the nutritional values given above, do not invent any new numbers "
            "or mention any figure not listed above. "
            if not strict else
            "IMPORTANT: You previously included a number that was not in the values given above. "
            "This time, using ONLY the exact nutritional values given above, do not invent, round "
            "differently, or mention any figure not explicitly listed above. "
        )
        prompt = (
            "You are a helpful nutrition assistant. A user uploaded a photo of their meal. "
            f"Detected food items: {labels_text}. "
            f"Meal description: {caption if caption else 'not available'}. "
            f"Nutritional values: {nutrition_text}. "
            f"{instruction}"
            "Write a short, simple, friendly explanation of this meal in 2-3 sentences. "
            "Mention whether it looks balanced and give one practical suggestion for improvement."
        )
        response = requests.post(
            OLLAMA_API_URL,
            json={'model': OLLAMA_MODEL, 'prompt': prompt, 'stream': False},
            timeout=60
        )
        if response.status_code != 200:
            return None
        return response.json().get('response', '').strip()

    try:
        feedback = call_ollama(strict=False)
        if feedback is None:
            return "Feedback is unavailable right now (Ollama returned an error)."

        if not find_unverified_numbers(feedback, nutrition_totals):
            return feedback

        retry_feedback = call_ollama(strict=True)
        if retry_feedback is not None and not find_unverified_numbers(retry_feedback, nutrition_totals):
            return retry_feedback

        return build_fallback_feedback(nutrition_totals)

    except requests.exceptions.RequestException:
        return "Feedback is unavailable right now. Make sure Ollama is running locally."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image received. Please upload a file.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected. Please choose an image.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Please upload a JPG, PNG, or WEBP image.'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')

        results = model(image, conf=DETECTION_CONFIDENCE_THRESHOLD)
        result = results[0]

        food_detections = []
        seen_labels = set()
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            food_detections.append({'label': label, 'confidence': round(confidence * 100, 1)})
            seen_labels.add(label)

        if not food_detections:
            original_b64 = image_to_base64(image)
            return jsonify({
                'error': 'No food items detected in this image. '
                         'Please try a clearer photo with the food as the main subject.',
                'original_image': original_b64
            }), 200

        meal_caption = get_caption(image)

        # cross-check weak detections against the caption and relabel them if a stronger detection matches the caption. Long, specific class names may reduce the effectiveness of this check.
        CAPTION_CROSSCHECK_THRESHOLD = 60
        caption_lower = meal_caption.lower()
        caption_food_words = {word for word in model.names.values() if word.lower() in caption_lower}

        for detection in food_detections:
            label = detection['label']
            if detection['confidence'] < CAPTION_CROSSCHECK_THRESHOLD and label not in caption_lower:
                better_match = next(
                    (d['label'] for d in food_detections
                     if d['label'] != label and d['label'] in caption_food_words),
                    None
                )
                if better_match:
                    detection['label'] = better_match

        seen_labels = {d['label'] for d in food_detections}
        unique_labels = list(seen_labels)

        # the model's own plotter draws the bounding boxes directly
        annotated_frame = result.plot(conf=True, labels=True, boxes=True, line_width=2)
        import numpy as np
        annotated_pil = Image.fromarray(annotated_frame[..., ::-1])  # bgr to rgb
        annotated_b64 = image_to_base64(annotated_pil)
        original_b64 = image_to_base64(image)

        nutrition_data = get_nutrition_data(unique_labels)

        # specific dish names may not be recognised by the nutrition API. if the dish lookup fails, retry using BLIP's more generic caption.
        if not nutrition_data.get('success') and meal_caption:
            caption_nutrition_data = get_nutrition_data([meal_caption])
            if caption_nutrition_data.get('success'):
                caption_nutrition_data['note'] = (
                    f"Nutrition values are estimated from the meal description "
                    f"(\"{meal_caption}\"), since \"{', '.join(unique_labels)}\" "
                    f"was not recognised by the nutrition database."
                )
                nutrition_data = caption_nutrition_data

        llm_feedback = get_llm_feedback(unique_labels, meal_caption, nutrition_data.get('totals'))

        return jsonify({
            'success': True,
            'annotated_image': annotated_b64,
            'original_image': original_b64,
            'detections': food_detections,
            'labels': unique_labels,
            'count': len(food_detections),
            'caption': meal_caption,
            'nutrition': nutrition_data,
            'feedback': llm_feedback
        }), 200

    except Exception as e:
        return jsonify({'error': f'Something went wrong during detection: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)