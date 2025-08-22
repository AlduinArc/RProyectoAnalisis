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
    path('graficos_avanzados/<int:id>/', views.graficos_avanzados, name='graficos_avanzados'),
    path('temp_analisis_grafico/<int:id>/', views.temp_analisis_grafico, name='temp_analisis_grafico'),
    path('modificar_separadores/<int:id>/', views.procesamiento_archivos, name='modificar_separadores'),
    path('modificar_separadores/vista/<int:id>/', views.vista_procesamiento_archivos, name='vista_modificar_separadores'),
    path('test_graficos/<int:id>/', views.Test_graficos, name='test_graficos'),  # gráficos de prueba
    
    path('interfaz_procesamiento/<int:id>/', views.interfaz_procesamiento, name='interfaz_procesamiento'),
    path('interfaz_modelado/<int:id>/', views.interfaz_modelado, name='interfaz_modelado'),
    
    path('eliminar_columnas_ceros/<int:id>/', views.eliminar_columnas_ceros, name='eliminar_columnas_ceros'),
    
    path('filtro_valores/<int:id>/', views.filtro_valores, name='filtro_valores'),
    path('aplicar_abs/<int:id>/', views.aplicar_abs, name='aplicar_abs'),
    path('aplicar_normalizacion/<int:id>/', views.aplicar_normalizacion, name='aplicar_normalizacion'),
    
    path('prueba_modelado_randomforest/<int:id>/', views.prueba_modelado_randomforest, name='prueba_modelado_randomforest'),
    path('descargar_predicciones/<int:id>/', views.descargar_predicciones, name='descargar_predicciones'),

    path('ver_columnas_nulas/<int:id>/', views.ver_columnas_nulas, name='ver_columnas_nulas'),
    path('confirmar_eliminacion_columnas_nulas/<int:id>/', views.confirmar_eliminacion_columnas_nulas, name='confirmar_eliminacion_columnas_nulas'),
    path('grafica/<int:id>/', views.ver_grafica, name='ver_grafica'),

    
]