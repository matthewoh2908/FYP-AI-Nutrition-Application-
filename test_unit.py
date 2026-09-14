# Unit tests for individual application functions
import stub_ml_deps  # Sets up fake ML dependencies for testing
import base64
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

import app


# allowed_file()
@pytest.mark.parametrize("filename", ["meal.jpg", "meal.JPG", "meal.jpeg", "meal.png", "meal.webp"])
def test_allowed_file_accepts_valid_extensions(filename):
    assert app.allowed_file(filename) is True


@pytest.mark.parametrize("filename", ["meal.gif", "meal.bmp", "meal.pdf", "meal.exe"])
def test_allowed_file_rejects_invalid_extensions(filename):
    assert app.allowed_file(filename) is False


def test_allowed_file_rejects_no_extension():
    assert app.allowed_file("meal") is False


# image_to_base64()
def test_image_to_base64_roundtrip():
    original = Image.new("RGB", (10, 10), color=(255, 0, 0))
    encoded = app.image_to_base64(original)

    # Should be valid base64 that decodes back into a readable JPEG
    decoded_bytes = base64.b64decode(encoded)
    decoded_image = Image.open(BytesIO(decoded_bytes))
    assert decoded_image.size == (10, 10)

# get_nutrition_data()
def test_get_nutrition_data_returns_early_for_empty_labels():
    result = app.get_nutrition_data([])
    assert result["success"] is False
    assert result["totals"] == {}


@patch("app.requests.get")
def test_get_nutrition_data_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"carbohydrates_total_g": 25, "fat_total_g": 0.3, "fiber_g": 4.4,
         "sugar_g": 19, "sodium_mg": 1, "potassium_mg": 422, "cholesterol_mg": 0}
    ]
    mock_get.return_value = mock_response

    result = app.get_nutrition_data(["rice"])

    assert result["success"] is True
    assert result["totals"]["carbohydrates_total_g"] == 25
    assert "calories" not in result["totals"]  # Premium field excluded


@patch("app.requests.get")
def test_get_nutrition_data_handles_api_error_status(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_get.return_value = mock_response

    result = app.get_nutrition_data(["rice"])

    assert result["success"] is False
    assert "401" in result["note"]


@patch("app.requests.get")
def test_get_nutrition_data_handles_network_failure(mock_get):
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError("no network")

    result = app.get_nutrition_data(["rice"])

    assert result["success"] is False
    assert "nutrition API" in result["note"]


# get_llm_feedback()
@patch("app.requests.post")
def test_get_llm_feedback_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "This meal looks balanced."}
    mock_post.return_value = mock_response

    feedback = app.get_llm_feedback(["rice"], "a bowl of rice", {"carbohydrates_total_g": 25})

    assert feedback == "This meal looks balanced."
    # Confirms that the prompt tells the LLM not to invent nutrition numbers
    sent_prompt = mock_post.call_args.kwargs["json"]["prompt"]
    assert "do not invent any new numbers" in sent_prompt
    assert mock_post.call_count == 1 


@patch("app.requests.post")
def test_get_llm_feedback_handles_ollama_not_running(mock_post):
    import requests
    mock_post.side_effect = requests.exceptions.ConnectionError("Ollama not running")

    feedback = app.get_llm_feedback(["rice"], "a bowl of rice", {})

    assert "Ollama" in feedback


# rule-based hallucination filter
def test_find_unverified_numbers_detects_fabricated_value():
    # Checks that fabricated nutrition values are detected
    real_totals = {"carbohydrates_total_g": 16.2, "fat_total_g": 13.4}
    text = "This meal has 16g carbs and 250mg of calcium, which is great."
    # 16g matches 16.2 within tolerance; 250mg does not match any real value
    unverified = app.find_unverified_numbers(text, real_totals)
    assert "250mg" in unverified
    assert not any("16" in u for u in unverified)


def test_find_unverified_numbers_allows_rounding_tolerance():
    real_totals = {"sodium_mg": 447.0}
    text = "This meal has roughly 447mg of sodium."
    assert app.find_unverified_numbers(text, real_totals) == []


def test_find_unverified_numbers_returns_empty_for_no_nutrition_data():
    assert app.find_unverified_numbers("This meal has 500g of something.", {}) == []


@patch("app.requests.post")
def test_get_llm_feedback_retries_once_then_succeeds(mock_post):
    bad_response = MagicMock()
    bad_response.status_code = 200
    bad_response.json.return_value = {"response": "This has 999g of an invented nutrient."}

    good_response = MagicMock()
    good_response.status_code = 200
    good_response.json.return_value = {"response": "This meal looks reasonably balanced."}

    mock_post.side_effect = [bad_response, good_response]

    feedback = app.get_llm_feedback(["rice"], "a bowl of rice", {"carbohydrates_total_g": 25})

    assert feedback == "This meal looks reasonably balanced."
    assert mock_post.call_count == 2  # Checks that the LLM retries once


@patch("app.requests.post")
def test_get_llm_feedback_falls_back_to_safe_template_after_failed_retry(mock_post):
    bad_response = MagicMock()
    bad_response.status_code = 200
    bad_response.json.return_value = {"response": "This has 999g of an invented nutrient."}
    mock_post.return_value = bad_response  

    nutrition_totals = {"carbohydrates_total_g": 16.2, "fat_total_g": 13.4,
                         "fiber_g": 17, "sodium_mg": 447}
    feedback = app.get_llm_feedback(["Bak Kut Teh"], "a bowl of soup", nutrition_totals)

    assert mock_post.call_count == 2
    assert "999" not in feedback  # Incorrect value is removed
    assert "16.2" in feedback  # Uses the actual nutrition value
