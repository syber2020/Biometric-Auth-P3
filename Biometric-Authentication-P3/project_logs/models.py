from __future__ import unicode_literals
from datetime import datetime
from django.db import models
from records.models import Records


class ProjectLogs(models.Model):
    id = models.AutoField(primary_key=True)
    Project_name = models.CharField(max_length=100, null=False)
    Role = models.CharField(max_length=100, null=False)
    Duration = models.CharField(max_length=100,null=False)
    Completion_percent = models.IntegerField()
    Deadline = models.DateField()

    def __str__(self):
        return self.id
    class Meta:
        verbose_name_plural = "ProjectLogs"
