import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from .models import ArchivoSubido
import os
from django.conf import settings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid
from django.templatetags.static import static


# Create your views here.
from .forms import ArchivoForm
def landing_page(request):
    return render(request, 'landing.html')

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
'''
def ver_archivo(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    filepath = archivo.archivo.path

    try:
        df = pd.read_csv(filepath, delimiter=';', on_bad_lines='skip')
        if df.columns.isnull().any():
            df.columns = [f"Columna_{i+1}" for i in range(df.shape[1])]

        # Crear gráfico con Matplotlib
        fig, ax = plt.subplots()
        df[df.columns[1]] = pd.to_numeric(df[df.columns[1]], errors='coerce')  # Por si hay errores
        df.plot(x=df.columns[0], y=df.columns[1], ax=ax)

        # Guardar la imagen temporalmente
        filename = f"{uuid.uuid4()}.png"
        ruta_imagen = os.path.join(settings.MEDIA_ROOT, filename)
        fig.savefig(ruta_imagen, bbox_inches='tight')
        plt.close(fig)

        url_imagen = settings.MEDIA_URL + filename

    except Exception as e:
        return render(request, 'ver_archivo.html', {'error': str(e)})

    return render(request, 'ver_archivo.html', {
        'df': df,
        'archivo': archivo,
        'grafico_url': url_imagen
    })
'''
def ver_archivo(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    filepath = archivo.archivo.path

    try:
        df = pd.read_csv(filepath, delimiter=';', on_bad_lines='skip')
        if df.columns.isnull().any():
            df.columns = [f"Columna_{i+1}" for i in range(df.shape[1])]

        # Contar nulos y ceros
        nulos = df.isnull().sum().sum()
        ceros = (df == 0).sum().sum()

        # Parámetros para graficar
        columnas = df.columns.tolist()
        x_col = request.GET.get('x', columnas[0])
        y_col = request.GET.get('y', columnas[1] if len(columnas) > 1 else columnas[0])

        # Crear gráfico
        fig, ax = plt.subplots()
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df.plot(x=x_col, y=y_col, ax=ax)

        filename = f"{uuid.uuid4()}.png"
        ruta_imagen = os.path.join(settings.MEDIA_ROOT, filename)
        fig.savefig(ruta_imagen, bbox_inches='tight')
        plt.close(fig)

        url_imagen = settings.MEDIA_URL + filename

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            tabla_html = df.to_html()

    except Exception as e:
        return render(request, 'ver_archivo.html', {'error': str(e), 'archivo': archivo})

    return render(request, 'ver_archivo.html', {
        'df': tabla_html,
        'archivo': archivo,
        'grafico_url': url_imagen,
        'nulos': nulos,
        'ceros': ceros,
        'columnas': columnas
    })


def eliminar_archivo(request, archivo_id):
    archivo = get_object_or_404(ArchivoSubido, pk=archivo_id)
    archivo_path = os.path.join(settings.MEDIA_ROOT, archivo.archivo.name)

    if os.path.exists(archivo_path):
        os.remove(archivo_path)

    archivo.delete()
    return redirect('listar_archivos')  # redirige a la vista de listado