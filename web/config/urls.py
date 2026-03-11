from django.contrib import admin
from django.urls import path
from platform_ui import views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', views.home, name="home"),
    path('analyze/', views.analyze_quality, name="analyze"),
    path('run/', views.run_pipeline, name="run_pipeline"),
    path('llm/tables/', views.llm_tables, name="llm_tables"),
]