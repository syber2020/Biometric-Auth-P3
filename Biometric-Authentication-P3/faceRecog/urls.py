
from django.conf.urls import url, include
from django.contrib import admin
from faceRecog import views as app_views
from django.contrib.auth import views

urlpatterns = [
    url(r'^user/(?P<username>\w+)/$', app_views.index, name='indexmain'),
    url(r'^admindashboard$', app_views.admindashboard),
    url(r'^empdashboard$', app_views.empdashboard),
    url(r'^error_image$', app_views.errorImg),
    url(r'^trainer$', app_views.trainer),
    url(r'^detect/(?P<username>\w+)/$', app_views.detect, name='detect'),
    url(r'^predictaudio$', app_views.predictaudio),
    url(r'^capture$', app_views.capture),
    url(r'^admin/', admin.site.urls),
    url(r'^records/', include('records.urls')),
]
