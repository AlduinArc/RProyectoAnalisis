from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('subir/', views.subir_archivo, name='subir_archivo'),
    path('listar/', views.listar_archivos, name='listar_archivos'),
    path('analizar/<int:archivo_id>/', views.analizar_archivo, name='analizar_archivo'),
    path('ver/<int:id>/', views.ver_archivo, name='ver_archivo'),
    path('eliminar/<int:archivo_id>/', views.eliminar_archivo, name='eliminar_archivo'),
    path('grafica/<int:id>/', views.ver_grafica, name='ver_grafica'),
    path('', views.landing_page, name='landing'),
]

##Not Found: /favicon.ico
##[23/Apr/2025 01:10:52,911] - Broken pipe from ('127.0.0.1', 50379)
##[23/Apr/2025 01:10:53] "GET /ver/7/ HTTP/1.1" 200 775145
##Not Found: /favicon.ico