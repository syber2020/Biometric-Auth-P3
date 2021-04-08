# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin


from .models import UserLogs
# admin.site.register(UserLogs)

@admin.register(UserLogs)
class UserLogsAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'login_time', 'confidence_score', 'login_status')

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False
