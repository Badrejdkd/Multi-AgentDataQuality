from django.contrib import admin
from django.urls import path
from platform_ui import views

urlpatterns = [
    path('admin/',           admin.site.urls),
    path('',                 views.dashboard,     name='dashboard'),
    path('extraction/',      views.extraction,    name='extraction'),
    path('extraction/run/',  views.run_extraction,name='run_extraction'),
    path('cleaning/',        views.cleaning,      name='cleaning'),
    path('cleaning/run/',    views.run_cleaning,  name='run_cleaning'),
    path('quality/',         views.quality,       name='quality'),
    path('quality/run/',     views.run_quality,   name='run_quality'),
    path('pipeline/',        views.pipeline,      name='pipeline'),
    path('pipeline/run/',    views.run_pipeline,  name='run_pipeline'),
    path('logs/',            views.logs_view,     name='logs'),
]