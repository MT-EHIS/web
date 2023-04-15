import pickle
from django.shortcuts import get_object_or_404
from sklearn.neural_network import MLPClassifier
from mtehis.model import LearningDetector
from mtehis.data import LearningDetectorData
from .models import Toolset


class ActiveToolset:
    def __init__(self, toolset: Toolset):
        detector_data: LearningDetectorData = pickle.loads(toolset.detector)
        self.detector = LearningDetector.create_from_data(detector_data)

        if toolset.classifier and len(toolset.classifier) > 0:
            self.classifier: MLPClassifier = pickle.loads(toolset.classifier)
        else:
            self.classifier = None


class ToolsetPool:
    _toolsets: dict[str, ActiveToolset] = {}

    @classmethod
    def fetch(cls):
        return cls._toolsets

    @classmethod
    def flush(cls):
        cls._toolsets = {}

    @classmethod
    def get_or_load_toolset(cls, name: str):
        if name not in cls._toolsets:
            cls._toolsets[name] = ActiveToolset(get_object_or_404(Toolset, name=name))
        return cls._toolsets[name]
