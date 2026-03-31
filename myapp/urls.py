from django.urls import path

from . import views

urlpatterns = [
    path("", views.start_training_view, name="home"),
    path("start_training/", views.start_training_view, name="start_training"),
    path("training_status_api/", views.training_status_api, name="training_status_api"),
    path("server_status/", views.server_status_view, name="server_status"),
]
