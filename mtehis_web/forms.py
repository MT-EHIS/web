import pickle
from django import forms
from django.core import validators
from sklearn.neural_network import MLPClassifier
from mtehis.model import LearningDetector
from mtehis.data import LearningDetectorData
from .models import Toolset


class ToolsetForm(forms.ModelForm):
    features_count = forms.IntegerField(min_value=1, label='Features count', widget=forms.NumberInput())

    classifier_hidden_layer_sizes = forms.CharField(label='Classifier hidden layer sizes',
                                                    help_text='comma separated values',
                                                    validators=[validators.validate_comma_separated_integer_list],
                                                    required=False)
    # TODO: add other parameters:
    # {
    # 'activation': 'relu',
    # 'alpha': 0.0001,
    # 'batch_size': 'auto',
    # 'beta_1': 0.9,
    # 'beta_2': 0.999,
    # 'early_stopping': False,
    # 'epsilon': 1e-08,
    # 'hidden_layer_sizes': (64, 32),
    # 'learning_rate': 'constant',
    # 'learning_rate_init': 0.001,
    # 'max_fun': 15000,
    # 'max_iter': 200,
    # 'momentum': 0.9,
    # 'n_iter_no_change': 10,
    # 'nesterovs_momentum': True,
    # 'power_t': 0.5,
    # 'random_state': None,
    # 'shuffle': True,
    # 'solver': 'adam',
    # 'tol': 0.0001,
    # 'validation_fraction': 0.1,
    # 'verbose': False,
    # 'warm_start': False,
    # }

    class Meta:
        model = Toolset
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        instance = kwargs.get('instance')

        if instance and instance.detector:
            learning_detector_data: LearningDetectorData = pickle.loads(instance.detector)
            self.fields['features_count'].initial = learning_detector_data.features_count

        if instance and instance.classifier and len(instance.classifier) > 0:
            classifier: MLPClassifier = pickle.loads(instance.classifier)
            classifier_params = classifier.get_params(True)
            hidden_layer_sizes_csv = ','.join(map(str, classifier_params['hidden_layer_sizes']))
            self.fields['classifier_hidden_layer_sizes'].initial = hidden_layer_sizes_csv

    def save(self, commit=True):
        features_count_value = self.cleaned_data.get('features_count')
        classifier_hidden_layer_sizes_value = self.cleaned_data.get('classifier_hidden_layer_sizes')

        if self.instance.detector is not None and len(self.instance.detector) > 0:
            tmp_learning_detector_data: LearningDetectorData = pickle.loads(self.instance.detector)
            tmp_learning_detector_data.features_count = features_count_value
            learning_detector = LearningDetector.create_from_data(tmp_learning_detector_data)
        else:
            learning_detector = LearningDetector(features_count_value)

        learning_detector_data = learning_detector.get_data()

        self.instance.detector = pickle.dumps(learning_detector_data)

        if 'classifier_hidden_layer_sizes' in self.changed_data:
            classifier_hidden_layer_sizes_values = tuple(map(int, classifier_hidden_layer_sizes_value.split(',')))
            classifier = MLPClassifier(hidden_layer_sizes=classifier_hidden_layer_sizes_values)

            self.instance.classifier = pickle.dumps(classifier)

        return super().save(commit=commit)
