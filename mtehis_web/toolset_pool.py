import pickle
import pandas as pd
from enum import Enum
from typing import Optional
from django.shortcuts import get_object_or_404
from sklearn.neural_network import MLPClassifier
from streamad.util import CustomDS
from mtehis.model import LearningDetector
from mtehis.data import LearningDetectorData
from mtehis.util import evaluate
from .models import Toolset


class EvaluationStatus(str, Enum):
    not_started = 'not started'
    processing = 'processing'
    finished = 'finished'


class EvaluationInfo:
    def __init__(self, train_ds: CustomDS = None, test_ds: CustomDS = None):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.status = EvaluationStatus.not_started
        self.iteration = 0
        self.result: Optional[pd.DataFrame] = None

    def start_evaluation(self, detector: LearningDetector, classifier: MLPClassifier = None):
        def _logging_func(i):
            self.iteration = i

        detector.logging_func = _logging_func

        self.status = EvaluationStatus.processing

        self.result = evaluate(detector, self.train_ds, self.test_ds)

        # TODO: classifier

        self.status = EvaluationStatus.finished

        return self.result


class ActiveToolset:
    def __init__(self, toolset: Toolset):
        self.id = toolset.id
        detector_data: LearningDetectorData = pickle.loads(toolset.detector)
        self.detector = LearningDetector.create_from_data(detector_data)

        if toolset.classifier and len(toolset.classifier) > 0:
            self.classifier: MLPClassifier = pickle.loads(toolset.classifier)
        else:
            self.classifier = None

        self.evaluation_info: Optional[EvaluationInfo] = None

    def save(self):
        toolset = Toolset.objects.get(id=self.id)
        toolset.detector = pickle.dumps(self.detector.get_data())
        if self.classifier:
            toolset.classifier = pickle.dumps(self.classifier)
        toolset.save()

    def start_evaluation(self, train_ds: CustomDS = None, test_ds: CustomDS = None, save_results=False):
        if self.evaluation_info is not None and self.evaluation_info.status == EvaluationStatus.processing:
            raise RuntimeError('Toolset is already being evaluated')

        self.evaluation_info = EvaluationInfo(train_ds, test_ds)

        result = self.evaluation_info.start_evaluation(self.detector, self.classifier)

        if save_results:
            self.save()

        return result


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
