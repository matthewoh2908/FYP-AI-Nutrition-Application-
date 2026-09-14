# This integration tests the /detect route end-to-end with simulated YOLO11, BLIP, nutrition API, and Ollama, while Flask routing, validation, and pipeline logic run in realtime.
import stub_ml_deps  # noqa: F401
import io
import numpy as np
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

import app


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client


def make_test_image_bytes():
    img = Image.new("RGB", (200, 200), color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def make_fake_box(class_id, confidence):
    box = MagicMock()
    box.cls = [class_id]
    box.conf = [confidence]
    return box


def make_fake_result(boxes, names):
    fake_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = MagicMock()
    result.boxes = boxes
    result.plot.return_value = fake_frame
    result.names = names
    return result


# Request validation

def test_detect_rejects_request_with_no_file(client):
    response = client.post("/detect", data={})
    assert response.status_code == 400
    assert "No image received" in response.get_json()["error"]


def test_detect_rejects_empty_filename(client):
    data = {"image": (io.BytesIO(b"fake"), "")}
    response = client.post("/detect", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "No file selected" in response.get_json()["error"]


def test_detect_rejects_disallowed_file_type(client):
    data = {"image": (io.BytesIO(b"fake"), "meal.gif")}
    response = client.post("/detect", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]


# No food detected path

@patch("app.model")
def test_detect_returns_friendly_error_when_nothing_found(mock_model, client):
    mock_model.return_value = [make_fake_result([], {})]

    data = {"image": (make_test_image_bytes(), "meal.jpg")}
    response = client.post("/detect", data=data, content_type="multipart/form-data")

    body = response.get_json()
    assert response.status_code == 200
    assert "No food items detected" in body["error"]
    assert "original_image" in body


# Full successful pipeline

@patch("app.get_llm_feedback")
@patch("app.get_nutrition_data")
@patch("app.get_caption")
@patch("app.model")
def test_detect_full_pipeline_success(mock_model, mock_caption, mock_nutrition, mock_feedback, client):
    class_names = {0: "rice"}
    mock_model.names = class_names
    mock_model.return_value = [make_fake_result([make_fake_box(0, 0.91)], class_names)]
    mock_caption.return_value = "a bowl of rice"
    mock_nutrition.return_value = {
        "success": True,
        "items": [{}],
        "totals": {"carbohydrates_total_g": 27, "fat_total_g": 0.4, "fiber_g": 3.1,
                   "sugar_g": 14, "sodium_mg": 1, "potassium_mg": 422, "cholesterol_mg": 0},
        "note": "Calories and protein are premium-only fields on API Ninjas and are not shown in this app."
    }
    mock_feedback.return_value = "This is a great source of potassium and fibre."

    data = {"image": (make_test_image_bytes(), "meal.jpg")}
    response = client.post("/detect", data=data, content_type="multipart/form-data")
    body = response.get_json()

    assert response.status_code == 200
    assert body["success"] is True
    assert body["labels"] == ["rice"]
    assert body["caption"] == "a bowl of rice"
    assert body["nutrition"]["totals"]["carbohydrates_total_g"] == 27
    assert body["feedback"] == "This is a great source of potassium and fibre."
    assert "annotated_image" in body and len(body["annotated_image"]) > 0


# Regression test for the caption cross-check

@patch("app.get_llm_feedback")
@patch("app.get_nutrition_data")
@patch("app.get_caption")
@patch("app.model")
def test_caption_crosscheck_fixes_low_confidence_mislabel(
    mock_model, mock_caption, mock_nutrition, mock_feedback, client
):

    # Regression test for the caption cross-check. A confident correct detection and a weak incorrect one are provided. Since BLIP mentions only the correct dish, the weak detection should be relabelled.
    class_names = {6: "sushi", 45: "sashimi"}
    mock_model.names = class_names
    mock_model.return_value = [make_fake_result([
        make_fake_box(6, 0.82),   # sushi, confident
        make_fake_box(45, 0.24),  # sashimi, weak/incorrect
    ], class_names)]
    mock_caption.return_value = "a plate of sushi on a wooden board"
    mock_nutrition.return_value = {"success": True, "items": [], "totals": {}, "note": ""}
    mock_feedback.return_value = "Looks like a light, protein-rich meal."

    data = {"image": (make_test_image_bytes(), "meal.jpg")}
    response = client.post("/detect", data=data, content_type="multipart/form-data")
    body = response.get_json()

    assert "sashimi" not in body["labels"]
    assert "sushi" in body["labels"]
    assert all(d["label"] == "sushi" for d in body["detections"])


# Nutrition caption-fallback

@patch("app.get_llm_feedback")
@patch("app.get_caption")
@patch("app.model")
@patch("app.requests.get")
def test_nutrition_falls_back_to_caption_when_dish_name_not_recognised(
    mock_requests_get, mock_model, mock_caption, mock_feedback, client
):
    # Regression test for nutrition lookup fallback. When the API does not recognise a specific dish name, the app should retry using BLIP's generic caption and explain the substitution in the response.
    class_names = {12: "Bak Kut Teh"}
    mock_model.names = class_names
    mock_model.return_value = [make_fake_result([make_fake_box(12, 0.71)], class_names)]
    mock_caption.return_value = "a bowl of soup with meat and vegetables"
    mock_feedback.return_value = "A hearty, protein-rich soup."

    dish_name_response = MagicMock()
    dish_name_response.status_code = 200
    dish_name_response.json.return_value = []  # Simulate no results for the specific dish name

    caption_response = MagicMock()
    caption_response.status_code = 200
    caption_response.json.return_value = [
        {"carbohydrates_total_g": 5, "fat_total_g": 12, "fiber_g": 1,
         "sugar_g": 0, "sodium_mg": 400, "potassium_mg": 300, "cholesterol_mg": 50}
    ]

    mock_requests_get.side_effect = [dish_name_response, caption_response]

    data = {"image": (make_test_image_bytes(), "meal.jpg")}
    response = client.post("/detect", data=data, content_type="multipart/form-data")
    body = response.get_json()

    assert response.status_code == 200
    assert mock_requests_get.call_count == 2  # Dish-name attempt, then caption fallback
    assert body["nutrition"]["success"] is True
    assert body["nutrition"]["totals"]["fat_total_g"] == 12
    assert "Bak Kut Teh" in body["nutrition"]["note"]
    assert "meal description" in body["nutrition"]["note"]
