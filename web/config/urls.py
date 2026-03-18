# platform_ui/urls.py - AJOUTER CES ROUTES

from django.contrib import admin
from django.urls import path
from platform_ui import views

urlpatterns = [
    # Routes existantes
    path('admin/', admin.site.urls),
    path('', views.dashboard, name='dashboard'),
    path('extraction/', views.extraction, name='extraction'),
    path('extraction/run/', views.run_extraction, name='run_extraction'),
    path('cleaning/', views.cleaning, name='cleaning'),
    path('cleaning/run/', views.run_cleaning, name='run_cleaning'),
    path('quality/', views.quality, name='quality'),
    path('quality/run/', views.run_quality, name='run_quality'),
    path('pipeline/', views.pipeline, name='pipeline'),
    path('pipeline/run/', views.run_pipeline, name='run_pipeline'),
    path('logs/', views.logs_view, name='logs'),
    
    # Routes pour filtrage et download
    path('data/filter/', views.filter_data, name='filter_data'),
    path('data/download/', views.download_csv, name='download_csv'),
    path('data/column-stats/', views.get_column_stats, name='column_stats'),
        # NOUVELLES ROUTES POUR LLM
    path('llm/select-tables/', views.llm_select_tables, name='llm_select_tables'),
    path('llm/generate-filters/', views.llm_generate_filters, name='llm_generate_filters'),
    path('llm/analyze-database/', views.llm_analyze_database, name='llm_analyze_database'),
    path('llm/', views.llm_interface, name='llm_interface'),  # Nouvelle page
 
]