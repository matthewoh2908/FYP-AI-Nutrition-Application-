# stubs heavy ML dependencies so app.py's non-ML logic can be tested without downloading model weights or installing the full ML stack.
import os
import sys
import types
from contextlib import contextmanager

# app.py requires API_NINJAS_KEY at import time, so tests set a dummy key. Nutrition API calls are mocked and never reach the real API. 
os.environ.setdefault("API_NINJAS_KEY", "test-key-not-a-real-key")
from unittest.mock import MagicMock

# torch test
fake_torch = types.ModuleType("torch")

@contextmanager
def _no_grad():
    yield

fake_torch.no_grad = _no_grad
sys.modules["torch"] = fake_torch

# BLIP test
fake_transformers = types.ModuleType("transformers")

class _FakePretrained:
    @classmethod
    def from_pretrained(cls, model_id):
        return MagicMock(name=f"{cls.__name__}({model_id})")

class BlipProcessor(_FakePretrained):
    pass

class BlipForConditionalGeneration(_FakePretrained):
    pass

fake_transformers.BlipProcessor = BlipProcessor
fake_transformers.BlipForConditionalGeneration = BlipForConditionalGeneration
sys.modules["transformers"] = fake_transformers

# YOLO test
fake_ultralytics = types.ModuleType("ultralytics")

class YOLO:
    def __init__(self, weights_path):
        self.weights_path = weights_path
    def __call__(self, *args, **kwargs):
        return [MagicMock(boxes=[])]

fake_ultralytics.YOLO = YOLO
sys.modules["ultralytics"] = fake_ultralytics