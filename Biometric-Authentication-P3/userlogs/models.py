# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from datetime import datetime
from django.db import models
from records.models import Records


class UserLogs(models.Model):
    id = models.AutoField(primary_key=True)
    user_ident = models.CharField(max_length=100, null=False, default='Vandana')
    user = models.ForeignKey(to=Records, on_delete=models.CASCADE)
    login_time = models.DateTimeField(null=False)
    confidence_score = models.CharField(max_length=100, null=False)
    login_status = models.CharField(max_length=100, null=False)

    def __str__(self):
        return self.id
    class Meta:
        verbose_name_plural = "UserLogs"
