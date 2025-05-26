import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import ArchivoSubido
from .forms import ArchivoForm
from django.template.loader import render_to_string
from django.http import JsonResponse
from io import BytesIO


def interfaz(request):
    archivos = ArchivoSubido.objects.all()
    return render(request, 'interfaz.html', {'archivos': archivos})



def landing_page(request):
    return render(request, 'landing.html')


def subir_archivo(request):
    if request.method == 'POST':
        form = ArchivoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    else:
        form = ArchivoForm()

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'partials/subir_contenido.html', {'form': form})
    return render(request, 'subir_archivo.html', {'form': form})


def listar_archivos(request):
    archivos = ArchivoSubido.objects.all()

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'partials/listar_contenido.html', {'archivos': archivos})
    return render(request, 'listar_archivos.html', {'archivos': archivos})


def analizar_archivo(request, archivo_id):
    archivo_obj = get_object_or_404(ArchivoSubido, id=archivo_id)
    ruta_completa = os.path.join(settings.MEDIA_ROOT, archivo_obj.archivo.name)

    if ruta_completa.endswith('.csv'):
        df = pd.read_csv(ruta_completa)
    elif ruta_completa.endswith('.xlsx'):
        df = pd.read_excel(ruta_completa)
    else:
        df = pd.DataFrame({'Error': ['Formato no soportado']})

    tabla_html = df.to_html(classes='table table-striped')

    contexto = {'tabla': tabla_html, 'archivo': archivo_obj}
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'partials/analizar_contenido.html', contexto)
    return render(request, 'analizar_archivo.html', contexto)


def ver_archivo(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    filepath = archivo.archivo.path

    try:
        df = pd.read_csv(filepath, delimiter=';', on_bad_lines='skip')
        if df.columns.isnull().any():
            df.columns = [f"Columna_{i+1}" for i in range(df.shape[1])]

        nulos = df.isnull().sum().sum()
        ceros = (df == 0).sum().sum()

        columnas = df.columns.tolist()
        x_col = request.GET.get('x', columnas[0])
        y_col = request.GET.get('y', columnas[1] if len(columnas) > 1 else columnas[0])

        fig, ax = plt.subplots()
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df_grafico = df[[x_col, y_col]].dropna()

        if not df_grafico.empty:
            df_grafico.plot(x=x_col, y=y_col, ax=ax)
        else:
            ax.text(0.5, 0.5, "No hay datos válidos para graficar",
                    ha='center', va='center', transform=ax.transAxes)

        filename = f"{uuid.uuid4()}.png"
        ruta_imagen = os.path.join(settings.MEDIA_ROOT, filename)
        fig.savefig(ruta_imagen, bbox_inches='tight')
        plt.close(fig)
        url_imagen = settings.MEDIA_URL + filename

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            tabla_html = df.to_html()

        contexto = {
            'df': tabla_html,
            'archivo': archivo,
            'grafico_url': url_imagen,
            'nulos': nulos,
            'ceros': ceros,
            'columnas': columnas
        }

        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return render(request, 'partials/ver_contenido.html', contexto)
        return render(request, 'ver_archivo.html', contexto)

    except Exception as e:
        return render(request, 'ver_grafica.html', {
            'archivo': archivo,
            'columnas': [],
            'error': str(e)
        })


def ver_grafica(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    df = pd.read_csv(archivo.archivo.path, sep=';', on_bad_lines='skip')
    columnas = df.columns.tolist()

    x = request.GET.get('x')
    y = request.GET.get('y')
    tipo = request.GET.get('tipo', 'line')

    grafico_url = None
    error = None

    try:
        if x and y and x in df.columns and y in df.columns:
            fig, ax = plt.subplots()
            if tipo == 'line':
                df.plot(x=x, y=y, ax=ax)
            elif tipo == 'bar':
                df.plot.bar(x=x, y=y, ax=ax)
            elif tipo == 'pie':
                df[y].value_counts().plot.pie(ax=ax, autopct='%1.1f%%')
            else:
                raise ValueError("Tipo de gráfico no válido")

            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close(fig)
            grafico_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
            buf.close()
    except Exception as e:
        error = str(e)

    context = {
        'archivo': archivo,
        'columnas': columnas,
        'grafico_url': grafico_url,
        'error': error,
        'x_seleccionada': x,
        'y_seleccionada': y,
        'tipo': tipo,
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        html = render_to_string('fragmento_grafica.html', context)
        return JsonResponse({'html': html})

    return render(request, 'ver_grafica.html', context)

def eliminar_archivo(request, archivo_id):
    archivo = get_object_or_404(ArchivoSubido, pk=archivo_id)
    archivo_path = os.path.join(settings.MEDIA_ROOT, archivo.archivo.name)

    if os.path.exists(archivo_path):
        os.remove(archivo_path)

    archivo.delete()
    return redirect('listar_archivos')
