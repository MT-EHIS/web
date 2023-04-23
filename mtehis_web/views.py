import json
import pickle
import numpy as np
import pandas as pd
from django.http import StreamingHttpResponse
from streamad.util import CustomDS
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .toolset_pool import ToolsetPool
from .models import Anomaly


@csrf_exempt
def detect_anomalies(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        if 'toolset' not in data or 'data' not in data:
            return JsonResponse({'status': 'error', 'message': 'Request body must have "toolset" and "data" fields'})

        toolset = ToolsetPool.get_or_load_toolset(data.get('toolset'))
        input_data = np.array(data.get('data'), dtype=np.float64)

        predicted, scores = toolset.detector.predict_with_scores(input_data)

        # TODO: classifier

        inputs_with_anomalies = input_data[predicted == 1]
        detector_scores_with_anomalies = scores[predicted == 1]

        anomalies_count = len(inputs_with_anomalies)
        for i in range(anomalies_count):
            inputs = inputs_with_anomalies[i].tolist()
            detector_scores = detector_scores_with_anomalies[i].tolist()
            Anomaly.objects.create(
                toolset_id=toolset.id,
                inputs=pickle.dumps(inputs),
                detector_scores=pickle.dumps(detector_scores),
                label=None,
            )

        return JsonResponse({'status': 'success', 'result': predicted.tolist()})

    elif request.method == 'GET':
        toolset_name = request.GET.get('toolset')
        datetime_from = request.GET.get('from', None)
        datetime_to = request.GET.get('to', None)

        query_filter = {'toolset__name': toolset_name}
        if datetime_from is not None:
            query_filter['created_at__gte'] = datetime_from
        if datetime_to is not None:
            query_filter['created_at__lte'] = datetime_to

        anomalies = Anomaly.objects.filter(**query_filter)

        result = list(anomalies.values())
        for item in result:
            item['inputs'] = pickle.loads(item['inputs'])
            item['detector_scores'] = pickle.loads(item['detector_scores'])
            if item['classifier_scores'] is not None and len(item['classifier_scores']) > 0:
                item['classifier_scores'] = pickle.loads(item['classifier_scores'])
            else:
                item['classifier_scores'] = None

        return JsonResponse({'status': 'success', 'result': result})

    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})


@csrf_exempt
@login_required
def train(request):
    if request.method == 'POST':
        train_file = request.FILES.get('train_ds', None)
        test_file = request.FILES.get('test_ds', None)
        toolset_name = request.POST.get('toolset')
        print_text = request.POST.get('print_text', False)

        train_df = pd.read_csv(train_file) if train_file else None
        test_df = pd.read_csv(test_file) if test_file else None

        def prepare_df(df: pd.DataFrame):
            if 'timestamp' in df.columns:
                df.set_index('timestamp')
                df.sort_values('timestamp')
            elif 'id' in df.columns:
                df.set_index('id')
            return df

        if train_df is not None:
            prepare_df(train_df)
        if test_df is not None:
            prepare_df(test_df)

        def update_generator():
            # noinspection PyTypeChecker
            train_dataset = CustomDS(train_df) if train_df is not None else None
            # noinspection PyTypeChecker
            test_dataset = CustomDS(test_df) if test_df is not None else None

            toolset = ToolsetPool.get_or_load_toolset(toolset_name)

            if print_text:
                yield 'Training started.\n'

            result = toolset.start_evaluation(train_dataset, test_dataset, save_results=True)

            if print_text:
                yield 'Detector data was saved.\n'

            yield result.to_json()

            detector_data = toolset.detector.get_data()

            yield detector_data.__repr__()

            # TODO: classifier

            if print_text:
                yield 'Processing complete.\n'

        response = StreamingHttpResponse(update_generator(), status=200, content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        return response

    elif request.method == 'GET':
        toolset_name = request.GET.get('toolset')
        ts = ToolsetPool.get_or_load_toolset(toolset_name)

        if ts.evaluation_info is None:
            return JsonResponse({'status': 'success', 'result': 'Training data not provided'})

        return JsonResponse({
            'status': 'success',
            'iteration': ts.evaluation_info.iteration,
            'evaluation': ts.evaluation_info.status,
            'result': ts.evaluation_info.result.to_json() if ts.evaluation_info.result is not None else None,
        })
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
