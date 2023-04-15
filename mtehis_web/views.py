import json
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .toolset_pool import ToolsetPool


@csrf_exempt
def detect_anomalies(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        if 'toolset' not in data or 'data' not in data:
            return JsonResponse({'status': 'error', 'message': 'Request body must have "toolset" and "data" fields'})

        toolset = ToolsetPool.get_or_load_toolset(data.get('toolset'))
        input_data = np.array(data.get('data'), dtype=np.float64)

        predicted = toolset.detector.predict(input_data)

        # TODO: classifier

        # TODO: save anomalies to DB

        return JsonResponse({'status': 'success', 'result': predicted.tolist()})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
