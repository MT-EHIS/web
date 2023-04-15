from django.contrib import admin
from .models import Toolset, AnomalyClass, AnomalyClassMapping
from .forms import ToolsetForm


class ToolsetAdmin(admin.ModelAdmin):
    form = ToolsetForm
    fieldsets = [
        (
            None,
            {
                'fields': ['name', 'features_count'],
            },
        ),
        (
            'Classifier options',
            {
                'classes': ['collapse'],
                'fields': ['classifier_hidden_layer_sizes'],
            },
        ),
    ]


class AnomalyClassMappingAdmin(admin.ModelAdmin):
    fields = ['toolset', ('index', 'anomaly_class')]


admin.site.register(Toolset, ToolsetAdmin)
admin.site.register(AnomalyClass)
admin.site.register(AnomalyClassMapping, AnomalyClassMappingAdmin)
