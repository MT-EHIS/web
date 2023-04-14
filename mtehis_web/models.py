from django.db import models
from django.urls import reverse


class Toolset(models.Model):

    # Fields
    name = models.CharField(max_length=32, help_text='Enter toolset short name', unique=True)

    detector = models.BinaryField(help_text='Serialized data for detector')

    classifier = models.BinaryField(help_text='Serialized data for classifier', null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    # Metadata
    class Meta:
        ordering = ['name', '-created_at']

    # Methods
    def get_absolute_url(self):
        return reverse('toolset', args=[str(self.id)])

    def __str__(self):
        return self.name


class AnomalyClass(models.Model):

    # Fields
    name = models.CharField(max_length=32, help_text='Enter anomaly class short name', unique=True)

    # Metadata
    class Meta:
        ordering = ['name']
        verbose_name_plural = 'Anomaly classes'

    # Methods
    def get_absolute_url(self):
        return reverse('anomaly-class', args=[str(self.id)])

    def __str__(self):
        return self.name


class AnomalyClassMapping(models.Model):

    # Fields
    toolset = models.ForeignKey(Toolset, on_delete=models.CASCADE)

    index = models.PositiveSmallIntegerField(help_text='Index of anomaly class detected by classifier')

    anomaly_class = models.ForeignKey(AnomalyClass, on_delete=models.SET_NULL, null=True)

    # Metadata
    class Meta:
        ordering = ['index', 'toolset']
        unique_together = [['toolset', 'index'], ['toolset', 'anomaly_class']]

    # Methods
    def get_absolute_url(self):
        return reverse('anomaly-class-mapping', args=[str(self.id)])

    def __str__(self):
        return self.toolset.name + ' #' + str(self.index) + ' -> ' + self.anomaly_class.name


class Anomaly(models.Model):

    # Fields
    toolset = models.ForeignKey(Toolset, on_delete=models.CASCADE)

    inputs = models.BinaryField()

    detector_scores = models.BinaryField()

    classifier_scores = models.BinaryField()

    label = models.ForeignKey(AnomalyClass, on_delete=models.SET_NULL, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    # Metadata
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Anomalies'

    # Methods
    def get_absolute_url(self):
        return reverse('anomaly', args=[str(self.id)])

    def __str__(self):
        return 'Anomaly: ' + str(self.label.name if self.label is not None else 'unknown')
