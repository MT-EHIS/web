from django.urls import path
from .views import detect_anomalies, train

urlpatterns = [
    path('detect', detect_anomalies, name='detect_anomalies'),
    path('train', train, name='train'),
]
