from __future__ import unicode_literals
from datetime import datetime
from django.db import models
from records.models import Records


class EmpResource(models.Model):
    id = models.AutoField(primary_key=True)
    emp_name = models.CharField(max_length=100, null=False)
    emp_id = models.IntegerField()
    casual_leave = models.IntegerField()
    total_leave = models.IntegerField()
    medical_leave = models.IntegerField()


    def __str__(self):
        return self.id
    class Meta:
        managed = True
        verbose_name_plural = "EmpResource"
