from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView  # Importa la vista de logout incorporada

urlpatterns = [
    # Página raíz redirigible (cambia a login o interfaz según autenticación)
    path('', views.landing_page, name='landing'),  # Página pública inicial
    
    # Autenticación
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.custom_logout, name='logout'),  # Vista de logout incorporada
    
    # Vistas protegidas
    path('dashboard/', views.interfaz, name='interfaz'),  # Mueve la interfaz a /dashboard/
    path('subir/', views.subir_archivo, name='subir_archivo'),
    path('listar/', views.listar_archivos, name='listar_archivos'),
    path('analizar/<int:archivo_id>/', views.analizar_archivo, name='analizar_archivo'),
    path('ver/<int:id>/', views.ver_archivo, name='ver_archivo'),
    path('eliminar/<int:archivo_id>/', views.eliminar_archivo, name='eliminar_archivo'),
    path('analisis_grafico/<int:id>/', views.analisis_grafico, name='analisis_grafico'),
    path('graficos_columnas/<int:id>/', views.ver_graficos_columnas, name='ver_graficos_columnas'),

    path('temp_analisis_grafico/<int:id>/', views.temp_analisis_grafico, name='temp_analisis_grafico'),
    path('grafica/<int:id>/', views.ver_grafica, name='ver_grafica'),
    
]