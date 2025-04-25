import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from .models import ArchivoSubido
import os
from django.conf import settings

# Create your views here.
from .forms import ArchivoForm

def subir_archivo(request):
    if request.method == 'POST':
        form = ArchivoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    else:
        form = ArchivoForm()

    return render(request, 'subir_archivo.html', {'form': form})

def analizar_archivo(request, archivo_id):
    archivo_obj = ArchivoSubido.objects.get(id=archivo_id)
    ruta_completa = os.path.join(settings.MEDIA_ROOT, archivo_obj.archivo.name)

    # Leer el archivo dependiendo del tipo
    if ruta_completa.endswith('.csv'):
        df = pd.read_csv(ruta_completa)
    elif ruta_completa.endswith('.xlsx'):
        df = pd.read_excel(ruta_completa)
    else:
        df = pd.DataFrame({'Error': ['Formato no soportado']})

    # Convertir el DataFrame a HTML para mostrarlo
    tabla_html = df.to_html(classes='table table-striped')

    return render(request, 'analizar_archivo.html', {'tabla': tabla_html, 'archivo': archivo_obj})

def listar_archivos(request):
    archivos = ArchivoSubido.objects.all()
    return render(request, 'listar_archivos.html', {'archivos': archivos})

def ver_archivo(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    filepath = archivo.archivo.path  # Ruta del archivo

    try:
        # Leer el archivo CSV y asegurarnos de que las columnas sean las correctas
        df = pd.read_csv(filepath, on_bad_lines='skip')

        # Si no hay nombres de columna, podemos agregar nombres por defecto
        if df.columns.isnull().any():
            df.columns = [f"Columna_{i+1}" for i in range(df.shape[1])]

    except Exception as e:
        # Si ocurre un error, mostrar el mensaje
        return render(request, 'ver_archivo.html', {'error': str(e)})

    return render(request, 'ver_archivo.html', {'df': df, 'archivo': archivo})

def eliminar_archivo(request, archivo_id):
    archivo = get_object_or_404(ArchivoSubido, pk=archivo_id)
    archivo_path = os.path.join(settings.MEDIA_ROOT, archivo.archivo.name)

    if os.path.exists(archivo_path):
        os.remove(archivo_path)

    archivo.delete()
    return redirect('listar_archivos')  # redirige a la vista de listado