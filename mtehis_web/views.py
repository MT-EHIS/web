import json
import numpy as np
import pandas as pd
from django.http import StreamingHttpResponse
from streamad.util import CustomDS
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from .toolset_pool import ToolsetPool


@require_POST
@csrf_exempt
def detect_anomalies(request):
    data = json.loads(request.body)

    if 'toolset' not in data or 'data' not in data:
        return JsonResponse({'status': 'error', 'message': 'Request body must have "toolset" and "data" fields'})

    toolset = ToolsetPool.get_or_load_toolset(data.get('toolset'))
    input_data = np.array(data.get('data'), dtype=np.float64)

    predicted = toolset.detector.predict(input_data)

    # TODO: classifier

    # TODO: save anomalies to DB

    return JsonResponse({'status': 'success', 'result': predicted.tolist()})


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
