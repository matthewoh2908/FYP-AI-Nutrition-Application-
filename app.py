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

# API keys and URLs
API_NINJAS_KEY = os.environ.get('API_NINJAS_KEY')
if not API_NINJAS_KEY:
    raise RuntimeError(
        "API_NINJAS_KEY environment variable is not set. "
        "Get a free key from https://api-ninjas.com and set it, "
        "e.g. on Windows: set API_NINJAS_KEY=your_key_here"
    )
NUTRITION_API_URL = 'https://api.api-ninjas.com/v1/nutrition'

# local Ollama LLM for personalised feedback
OLLAMA_API_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'llama3'

# YOLO model for food detection. This is a custom-trained model based on YOLOv8, trained on a dataset of 1000+ food images with 50+ classes. 

# Best trained model
model = YOLO('models/best.pt')

# BLIP model for image captioning. 
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

# Only keep detections with at least 25% confidence
DETECTION_CONFIDENCE_THRESHOLD = 0.25

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(pil_image):
    # Converts a PIL image to a base64-encoded string for embedding in HTML
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def get_caption(pil_image):
    # Generates a caption for the given PIL image using the BLIP model. Returns an empty string if captioning fails.
    try:
        inputs = blip_processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            output_ids = blip_model.generate(**inputs, max_new_tokens=30)
        return blip_processor.decode(output_ids[0], skip_special_tokens=True)
    except Exception:
        return ""


def get_nutrition_data(food_labels):
    # Queries the API Ninjas nutrition API for the given list of food labels. Returns a dictionary with success status, items, totals, and a note.
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
    # Checks the feedback text for any numbers (in grams or milligrams) that do not match the real nutrition values. Returns a list of unverified numbers found in the text.
    if not nutrition_totals:
        return []

    real_values = [round(v, 1) for v in nutrition_totals.values() if isinstance(v, (int, float))]

    unverified = []
    for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(g|mg)\b', feedback_text, re.IGNORECASE):
        claimed_value = float(match.group(1))
        # Allow a small tolerance of 0.5 for rounding differences
        if not any(abs(claimed_value - real) <= 0.5 for real in real_values):
            unverified.append(match.group(0))

    return unverified


def build_fallback_feedback(nutrition_totals):
   # Builds a safe fallback feedback message using the real nutrition values, in case the LLM-generated feedback contains unverified numbers.
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
    # Generates feedback text using the Ollama LLM, based on the detected food labels, meal caption, and nutrition totals. If the LLM generates unverified numbers, it retries with stricter instructions. If that fails, it falls back to a safe message
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

        # Cross-check the detected food labels with the caption to correct any low-confidence detections
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

        # Generate an annotated image with bounding boxes and labels
        annotated_frame = result.plot(conf=True, labels=True, boxes=True, line_width=2)
        import numpy as np
        annotated_pil = Image.fromarray(annotated_frame[..., ::-1])  # bgr to rgb
        annotated_b64 = image_to_base64(annotated_pil)
        original_b64 = image_to_base64(image)

        nutrition_data = get_nutrition_data(unique_labels)

        # If the nutrition API did not return data for the detected labels, try using the caption to get nutrition data
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
