# Standard Library
import os
import io
import uuid
import mimetypes
from io import BytesIO

# Data Processing
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Configuración para evitar conflictos con GUI
import matplotlib.pyplot as plt
from PIL import Image  # Para procesamiento de imágenes
import base64
import json
import plotly.graph_objs as go
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Django modeling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Django Core
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.http import JsonResponse, FileResponse, HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.template.loader import render_to_string
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login
from django.views.decorators.cache import never_cache
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.views.decorators.csrf import csrf_protect
from django.db import IntegrityError
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.views.decorators.http import require_POST

# App Specific
from .models import ArchivoSubido
from .forms import ArchivoForm
from django.contrib.auth.decorators import user_passes_test
from django.contrib.auth import logout


def superuser_check(user):
    return user.is_superuser

@user_passes_test(superuser_check)
def vista_exclusiva_superuser(request):
    # Vista solo para superusuarios
    return render(request, 'admin/superuser_panel.html')
""""
@csrf_protect
def login_views(request):
    error = None
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect('interfaz')  # Redirige a tu página principal
        else:
            error = "Usuario o contraseña incorrectos"
    return render(request, 'login.html', {'error': error})
"""""
@csrf_protect

def login_view(request):
    if request.user.is_authenticated:
        return redirect('interfaz')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            auth_login(request, user)
            next_url = request.POST.get('next', 'interfaz')  # Redirige a 'next' o dashboard
            return redirect(next_url)
        else:
            error = "Credenciales inválidas"
            return render(request, 'login.html', {'error': error})
    
    return render(request, 'login.html')

@login_required
@never_cache

def interfaz(request):
    archivos = ArchivoSubido.objects.all()
    return render(request, 'interfaz.html', {'archivos': archivos})

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def clean_username(self):
        username = self.cleaned_data['username']
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("Este nombre de usuario ya está en uso.")
        return username

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user

def landing_page(request):
    return render(request, 'landing.html')

@login_required
def subir_archivo(request):
    if request.method == 'POST':
        form = ArchivoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "Archivo subido exitosamente")  # <-- Añade esta línea
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': True})  # Opcional para AJAX
            return redirect('interfaz')  # Redirige a donde necesites
            
    else:
        form = ArchivoForm()

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'partials/subir_contenido.html', {'form': form})
    return render(request, 'subir_archivo.html', {'form': form})

@login_required
def listar_archivos(request):
    archivos = ArchivoSubido.objects.all()

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'partials/listar_contenido.html', {'archivos': archivos})
    return render(request, 'listar_archivos.html', {'archivos': archivos})

@require_POST  # Asegura que solo se pueda llamar via POST
def custom_logout(request):
    logout(request)
    request.session.flush()
    response = redirect('landing')  # Redirige a tu página landing
    response.delete_cookie('sessionid')
    response.delete_cookie('csrftoken')
    return response

@login_required
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

@login_required
def ver_archivo(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    filepath = archivo.archivo.path
    extension = os.path.splitext(filepath)[1].lower()

    if extension in ['.csv', '.xlsx', '.xls']:
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
                ax.text(0.5, 0.5, "No hay datos válidos para graficar", ha='center', va='center', transform=ax.transAxes)

            filename = f"{uuid.uuid4()}.png"
            ruta_imagen = os.path.join(settings.MEDIA_ROOT, filename)
            fig.savefig(ruta_imagen, bbox_inches='tight')
            plt.close(fig)
            url_imagen = settings.MEDIA_URL + filename

            # Aquí siempre generamos el análisis descriptivo
            # Filtrar solo columnas numéricas
            df_numerico = df.select_dtypes(include='number')
            descripcion = df_numerico.describe()


            # Filtrar si se solicita ocultar columnas con count = 0.0
            if request.GET.get('ocultar') == '1':
                descripcion = descripcion.loc[:, descripcion.loc['count'] != 0.0]

            descripcion_html = descripcion.fillna('').to_html(
                classes="table table-striped table-bordered"
            )


            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                if request.GET.get('completo') == '1':
                    tabla_html = df.to_html(classes="table table-striped table-bordered")
                else:
                    tabla_html = df.head(7).to_html(classes="table table-striped table-bordered")

            contexto = {
                'df': tabla_html,
                'descripcion': descripcion_html,
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


@login_required
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
            fig, ax = plt.subplots(figsize=(10, 6))  # mejor tamaño
            
            # Preprocesar columnas
            df = df[[x, y]].dropna()

            # Convertir Y a numérico si es posible
            df[y] = pd.to_numeric(df[y], errors='coerce')
            
            if tipo == 'line':
                df_sorted = df.sort_values(by=x)  # ordenar para gráfico de línea
                ax.plot(df_sorted[x], df_sorted[y], marker='o')
                ax.set_title(f'Tendencia: {y} vs {x}')
                ax.set_xlabel(x)
                ax.set_ylabel(y)

            elif tipo == 'bar':
                # Si x tiene pocos valores únicos, usar groupby
                if df[x].nunique() < 30:
                    df_grouped = df.groupby(x)[y].mean().reset_index()
                    ax.bar(df_grouped[x], df_grouped[y])
                    ax.set_title(f'Bar chart: {y} promedio por {x}')
                    ax.set_xlabel(x)
                    ax.set_ylabel(f'{y} (promedio)')
                else:
                    raise ValueError("Demasiados valores únicos para gráfico de barras. Elija otra columna para X.")

            elif tipo == 'pie':
                # Pie chart solo si Y es categórica o pocos valores únicos
                if df[y].nunique() < 20:
                    df_counts = df[y].value_counts()
                    df_counts.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
                    ax.set_ylabel('')
                    ax.set_title(f'Pie chart de {y}')
                else:
                    raise ValueError("Gráfico de torta solo disponible para columnas con pocos valores únicos.")
            
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
        html = render_to_string('partials/fragmento_grafica.html', context)
        return JsonResponse({'html': html})

    return render(request, 'ver_grafica.html', context)


@login_required
def analisis_grafico(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    filepath = archivo.archivo.path
    extension = os.path.splitext(filepath)[1].lower()

    if extension in ['.csv', '.xlsx', '.xls']:
        try:
            df = pd.read_csv(filepath, delimiter=';', on_bad_lines='skip')

            # Eliminar fila final si es completamente NaN
            if df.tail(1).isnull().all(axis=1).any():
                df = df.iloc[:-1]

            # ✅ Filtrar columnas eliminando las que son texto puro (object)
            columnas_texto = df.select_dtypes(include=['object', 'string']).columns.tolist()
            columnas_permitidas = [col for col in df.columns if col not in columnas_texto]

            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                columnas_seleccionadas = request.GET.getlist('columnas')
                tipo = request.GET.get('tipo')

                graficos = []

                for col in columnas_seleccionadas:
                    if col not in df.columns:
                        continue
                    
                    fig, ax = plt.subplots()
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    if tipo == 'hist':
                        df[col].dropna().plot.hist(ax=ax, bins=30, color='skyblue')
                        ax.set_title(f"Histograma de {col}")
                    elif tipo == 'box':
                        df[[col]].dropna().plot.box(ax=ax)
                        ax.set_title(f"Boxplot de {col}")
                    elif tipo == 'pie':
                        conteo = df[col].value_counts().head(10)
                        ax.pie(conteo, labels=conteo.index, autopct='%1.1f%%')
                        ax.set_title(f"Gráfico de torta de {col}")
                        ax.set_ylabel("")
                    else:
                        continue

                    buf = BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png')
                    plt.close(fig)
                    imagen_base64 = base64.b64encode(buf.getvalue()).decode()
                    buf.close()

                    graficos.append({
                        'columna': col,
                        'imagen': f'data:image/png;base64,{imagen_base64}'
                    })

                html = render_to_string('partials/fragmento_analisis_grafica.html', {'graficos': graficos})
                return JsonResponse({'html': html})

            # Si no es AJAX, carga solo la página base con el formulario (solo columnas permitidas)
            contexto = {
                'archivo': archivo,
                'columnas': columnas_permitidas  # ✔️ Solo columnas filtradas
            }
            return render(request, 'ver_analisis_grafico.html', contexto)

        except Exception as e:
            return render(request, 'ver_analisis_grafico.html', {
                'archivo': archivo,
                'columnas': [],
                'error': str(e)
            })


"""
def analisis_grafico(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    df = pd.read_csv(archivo.archivo.path, sep=';', on_bad_lines='skip')
    columnas = df.columns.tolist()

    x = request.GET.get('x')
    y = request.GET.get('y')
    tipo = request.GET.get('tipo', 'scatter')

    grafico_url = None
    error = None

    try:
        if x and y and x in df.columns and y in df.columns:
            fig, ax = plt.subplots()

            # Gráfico de dispersión (scatter plot)
            if tipo == 'scatter':
                df.plot.scatter(x=x, y=y, ax=ax)
            elif tipo == 'hist':
                df[y].plot.hist(ax=ax, bins=30)
            elif tipo == 'box':
                df[[y]].plot.box(ax=ax)
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
        html = render_to_string('partials/fragmento_analisis_grafica.html', context)
        return JsonResponse({'html': html})

    return render(request, 'ver_analisis_grafico.html', context)
"""
@login_required 
def temp_analisis_grafico(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    df = pd.read_csv(archivo.archivo.path, sep=';', on_bad_lines='skip')

    # ✅ Filtrar columnas eliminando las que son texto puro (object)
    columnas_texto = df.select_dtypes(include=['object', 'string']).columns.tolist()
    columnas_permitidas = [col for col in df.columns if col not in columnas_texto]

    x = request.GET.get('x')
    ys = request.GET.getlist('y')  # ✔️ Capturar múltiples Y
    tipo = request.GET.get('tipo', 'line')

    graficos = []
    error = None

    try:
        if not ys or not x:
            raise ValueError("Debe seleccionar un eje X y al menos una columna Y.")

        for y in ys:
            if y not in df.columns or x not in df.columns:
                continue

            fig, ax = plt.subplots()
            fig.set_size_inches(6, 4)  # ✅ Ajuste de tamaño

            # Convertir a numérico solo si no es fecha (para seguridad adicional)
            df[x] = pd.to_numeric(df[x], errors='coerce')
            df[y] = pd.to_numeric(df[y], errors='coerce')

            df_filtrado = df.dropna(subset=[x, y])

            if tipo == 'line':
                df_filtrado = df_filtrado.sort_values(by=x)
                ax.plot(df_filtrado[x], df_filtrado[y], marker='o')
                ax.set_title(f'Tendencia: {y} sobre {x}')
            elif tipo == 'scatter':
                ax.scatter(df_filtrado[x], df_filtrado[y])
                ax.set_title(f'Dispersión: {y} vs {x}')
            elif tipo == 'bar':
                resumen = df_filtrado.groupby(x)[y].mean().sort_values(ascending=False).head(15)
                resumen.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'Promedio de {y} por {x}')
                ax.set_ylabel(f'{y} (promedio)')
                ax.set_xlabel(x)
            else:
                continue  # Si el tipo no es válido, ignora

            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)  # ✔️ DPI adicional
            plt.close(fig)
            grafico_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
            buf.close()

            graficos.append(grafico_url)

        if not graficos:
            raise ValueError("No se generaron gráficos. Verifique los datos.")

    except Exception as e:
        error = str(e)

    context = {
        'archivo': archivo,
        'columnas': columnas_permitidas,  # ✔️ Solo columnas numéricas
        'graficos': graficos,
        'error': error,
        'x_seleccionada': x,
        'tipo': tipo,
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        html = render_to_string('partials/fragmento_grafica.html', context)
        return JsonResponse({'html': html})

    return render(request, 'temp_ver_analisis_grafico.html', context)


@login_required
def comparacion_graficos(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    df = pd.read_csv(archivo.archivo.path, sep=';', on_bad_lines='skip')
    columnas = df.columns.tolist()

    x1 = request.GET.get('x1')
    y1 = request.GET.get('y1')
    tipo1 = request.GET.get('tipo1', 'line')

    x2 = request.GET.get('x2')
    y2 = request.GET.get('y2')
    tipo2 = request.GET.get('tipo2', 'bar')

    graficos = []
    error = None

    try:
        for x, y, tipo in [(x1, y1, tipo1), (x2, y2, tipo2)]:
            fig, ax = plt.subplots()
            if tipo == 'line':
                df.plot.line(x=x, y=y, ax=ax)
            elif tipo == 'bar':
                df.dropna(subset=[x, y]).groupby(x)[y].mean().plot(kind='bar', ax=ax)
            elif tipo == 'box':
                df[[y]].dropna().plot.box(ax=ax)
            elif tipo == 'pie':
                conteo = df[y].value_counts().head(10)
                ax.pie(conteo, labels=conteo.index, autopct='%1.1f%%')
                ax.set_ylabel("")
            else:
                raise ValueError("Tipo de gráfico no válido")

            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close(fig)
            img_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
            buf.close()
            graficos.append(img_url)

    except Exception as e:
        error = str(e)

    context = {
        'archivo': archivo,
        'columnas': columnas,
        'grafico1': graficos[0] if len(graficos) > 0 else None,
        'grafico2': graficos[1] if len(graficos) > 1 else None,
        'x1': x1, 'y1': y1, 'tipo1': tipo1,
        'x2': x2, 'y2': y2, 'tipo2': tipo2,
        'error': error,
    }

    return render(request, 'comparacion_graficos.html', context)


@login_required
def graficos_avanzados(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    df = pd.read_csv(archivo.archivo.path, sep=';', on_bad_lines='skip')
    
    # ✅ Filtrar columnas para quitar las de texto (object/string)
    columnas_texto = df.select_dtypes(include=['object', 'string']).columns.tolist()
    columnas_permitidas = [col for col in df.columns if col not in columnas_texto]

    modo = request.GET.get('modo')  # descripcion, comparacion, tendencia
    x = request.GET.get('x')
    y1 = request.GET.get('y1')
    y2 = request.GET.get('y2')

    grafico_url = None
    error = None

    try:
        fig, ax = plt.subplots()

        if modo == 'descripcion' and y1 in df.columns:
            df[y1] = pd.to_numeric(df[y1], errors='coerce')
            df[y1].dropna().plot.hist(ax=ax, bins=30, color='skyblue')
            ax.set_title(f"Histograma de {y1}")

        elif modo == 'comparacion' and x in df.columns and y1 in df.columns and y2 in df.columns:
            df[x] = pd.to_numeric(df[x], errors='coerce')
            df[y1] = pd.to_numeric(df[y1], errors='coerce')
            df[y2] = pd.to_numeric(df[y2], errors='coerce')
            df.dropna(subset=[x, y1, y2]).plot(x=x, y=[y1, y2], ax=ax)
            ax.set_title(f"Comparación entre {y1} y {y2} respecto a {x}")

        elif modo == 'boxplot' and y1 in df.columns:
            df[y1] = pd.to_numeric(df[y1], errors='coerce')
            df[[y1]].dropna().plot.box(ax=ax)
            ax.set_title(f"Boxplot de {y1}")

        elif modo == 'tendencia' and x in df.columns and y1 in df.columns and y2 in df.columns:
            df[x] = pd.to_datetime(df[x], errors='coerce')
            df[y1] = pd.to_numeric(df[y1], errors='coerce')
            df[y2] = pd.to_numeric(df[y2], errors='coerce')
            df = df.dropna(subset=[x, y1, y2]).sort_values(x)
            ax.plot(df[x], df[y1], label=y1)
            ax.plot(df[x], df[y2], label=y2)
            ax.set_title(f"Tendencia: {y1} vs {y2} sobre {x}")
            ax.legend()

        else:
            raise ValueError("Parámetros insuficientes o incorrectos")

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
        'columnas': columnas_permitidas,  # ✔️ Solo columnas numéricas o convertibles
        'grafico_url': grafico_url,
        'error': error,
        'modo': modo,
        'x': x,
        'y1': y1,
        'y2': y2
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        html = render_to_string('partials/fragmento_grafico_avanzado.html', context)
        return JsonResponse({'html': html})

    return render(request, 'graficos_avanzados.html', context)

@login_required
def vista_procesamiento_archivos(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    return render(request, 'modificar_separadores.html', {'archivo': archivo})


@login_required
def procesamiento_archivos(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    filepath = archivo.archivo.path

    try:
        df = pd.read_csv(filepath, sep=';', on_bad_lines='skip', dtype=str)  # Todo como texto
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)  # Reemplazar comas por puntos
        df.to_csv(filepath, sep=';', index=False)  # Sobrescribe el archivo
        return JsonResponse({'success': True, 'message': 'Archivo modificado correctamente.'})
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})

@login_required
def ver_graficos_columnas(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    filepath = archivo.archivo.path
    df = pd.read_csv(filepath, delimiter=';', on_bad_lines='skip')

    # Eliminar fila completamente vacía (si existe)
    if df.tail(1).isnull().all(axis=1).any():
        df = df.iloc[:-1]

    columnas = request.GET.getlist('columnas')
    imagenes_urls = []

    for col in columnas:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir a numérico si se puede
            fig, ax = plt.subplots()
            df[col].dropna().plot(kind='hist', ax=ax, bins=20, title=f'Distribución de {col}', color='skyblue')
            filename = f"{uuid.uuid4()}.png"
            path = os.path.join(settings.MEDIA_ROOT, filename)
            fig.savefig(path)
            plt.close(fig)
            imagenes_urls.append(settings.MEDIA_URL + filename)
        except Exception as e:
            print(f"Error procesando la columna {col}: {e}")

    return render(request, 'partials/graficos_multiples.html', {
        'imagenes': imagenes_urls
    })

@login_required
def ver_columnas_nulas(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    ruta = os.path.join(settings.MEDIA_ROOT, archivo.archivo.name)
    
    df = pd.read_csv(ruta)
    columnas_nulas = df.columns[df.isnull().any()].tolist()

    conteo_nulos = df.isnull().sum()
    conteo_nulos = conteo_nulos[conteo_nulos > 0]

    grafico_base64 = None

    if not conteo_nulos.empty:
        fig, ax = plt.subplots()
        conteo_nulos.plot(kind='bar', ax=ax, color='orange')
        ax.set_title('Valores nulos por columna')
        ax.set_ylabel('Cantidad de NaN')
        plt.xticks(rotation=45, ha='right')

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        grafico_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

    context = {
        'archivo_id': id,
        'columnas_nulas': columnas_nulas,
        'grafico': f"data:image/png;base64,{grafico_base64}" if grafico_base64 else None,
    }

    return render(request, 'ver_columnas_nulas.html', context)



@require_POST
def confirmar_eliminacion_columnas_nulas(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    ruta = os.path.join(settings.MEDIA_ROOT, archivo.archivo.name)

    df = pd.read_csv(ruta)

    # Eliminar columnas con al menos un NaN usando NumPy
    columnas_con_nan = df.columns[df.isnull().any()]
    df = df.drop(columns=columnas_con_nan)

    df.to_csv(ruta, index=False)  # Guardar los cambios

    return redirect('ver_archivo', archivo_id=id)

@login_required
def eliminar_columnas_ceros(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    ruta = archivo.archivo.path
    df = pd.read_csv(ruta, delimiter=';', on_bad_lines='skip')

    # Guardamos el dataframe original (para graficar antes)
    df_original = df.copy()

    # Contamos las columnas con solo ceros
    columnas_con_ceros = [col for col in df.columns if df[col].dtype != 'object' and (df[col] == 0).all()]
    cantidad_columnas_eliminadas = len(columnas_con_ceros)

    # Eliminamos esas columnas
    df.drop(columns=columnas_con_ceros, inplace=True)

    # Sobrescribimos el archivo CSV (eliminación permanente)
    df.to_csv(ruta, index=False, sep=';')

    # Generamos gráfico de tendencia antes y después
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=[df_original.shape[1]],
        x=["Antes"],
        name="Columnas antes",
        mode="lines+markers",
        line=dict(color="red")
    ))

    fig.add_trace(go.Scatter(
        y=[df.shape[1]],
        x=["Después"],
        name="Columnas después",
        mode="lines+markers",
        line=dict(color="green")
    ))

    grafico_html = fig.to_html(full_html=False)

    return render(request, 'eliminar_columnas_ceros.html', {
        'archivo': archivo,
        'cant_columnas_eliminadas': cantidad_columnas_eliminadas,
        'grafico_html': grafico_html
    })

@login_required
def Test_graficos(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    df = pd.read_csv(archivo.archivo.path, sep=';', on_bad_lines='skip')

    # ✅ Filtrar columnas numéricas
    columnas_texto = df.select_dtypes(include=['object', 'string']).columns.tolist()
    columnas_permitidas = [col for col in df.columns if col not in columnas_texto]

    x = request.GET.get('x')
    ys = request.GET.getlist('y')
    tipo = request.GET.get('tipo', 'line')

    graficos = []
    error = None

    if x and ys:
        try:
            for y in ys:
                if y not in df.columns or x not in df.columns:
                    continue

                fig, ax = plt.subplots()
                fig.set_size_inches(6, 4)

                # Asegurar que son numéricos
                df[x] = pd.to_numeric(df[x], errors='coerce')
                df[y] = pd.to_numeric(df[y], errors='coerce')
                df_filtrado = df.dropna(subset=[x, y])

                if tipo == 'line':
                    df_filtrado = df_filtrado.sort_values(by=x)
                    ax.plot(df_filtrado[x], df_filtrado[y], marker='o')
                    ax.set_title(f'Tendencia: {y} sobre {x}')
                elif tipo == 'scatter':
                    ax.scatter(df_filtrado[x], df_filtrado[y])
                    ax.set_title(f'Dispersión: {y} vs {x}')
                elif tipo == 'bar':
                    resumen = df_filtrado.groupby(x)[y].mean().sort_values(ascending=False).head(15)
                    resumen.plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title(f'Promedio de {y} por {x}')
                    ax.set_ylabel(f'{y} (promedio)')
                    ax.set_xlabel(x)
                else:
                    continue

                plt.tight_layout()
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                grafico_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
                buf.close()
                plt.close(fig)

                graficos.append(grafico_url)

            if not graficos:
                raise ValueError("No se generaron gráficos. Verifique los datos seleccionados.")

        except Exception as e:
            error = str(e)

    context = {
        'graficos': graficos,
        'error': error,
    }

    # ✅ AJAX: devolver solo el fragmento HTML para insertar en el modal
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        html = render_to_string('partials/fragmento_test.html', context)
        return JsonResponse({'html': html})

    # ✅ Render normal (si se carga directamente la URL sin AJAX)
    context.update({
        'archivo': archivo,
        'columnas': columnas_permitidas,
        'x_seleccionada': x,
        'tipo': tipo,
    })
    return render(request, 'Test_graficos.html', context)


# Función auxiliar para guardar nuevo archivo
def guardar_archivo_modificado(df, archivo, sufijo):
    import pandas as pd
    import os
    from django.conf import settings

    nuevo_nombre = f"{os.path.splitext(archivo.nombre)[0]}_{sufijo}.csv"
    nueva_ruta_absoluta = os.path.join(settings.MEDIA_ROOT, 'uploads', nuevo_nombre)
    
    # Usa el mismo delimitador que el archivo original (probablemente ; o ,)
    df.to_csv(nueva_ruta_absoluta, index=False, sep=';')  # <== usa sep=';' si así fue leído

    return nuevo_nombre




def filtro_valores(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    ruta = archivo.archivo.path
    df = pd.read_csv(ruta, sep=None, engine='python', on_bad_lines='skip')

    columnas = request.POST.getlist("columnas")

    valor = request.POST.get('valor')
    try:
        valor = float(valor)
        df_filtrado = df[df[columnas] != valor]
        nuevo = guardar_archivo_modificado(df_filtrado, archivo, f"filtro_{columnas}_{valor}")
        return HttpResponseRedirect(reverse('interfaz_procesamiento', args=[id]))
    except:
        return HttpResponseRedirect(reverse('interfaz_procesamiento', args=[id]))


def aplicar_abs(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    ruta = archivo.archivo.path
    df = pd.read_csv(ruta, sep=None, engine='python', on_bad_lines='skip')

    columnas = request.POST.getlist("columnas")

    df[columnas] = df[columnas].abs()
    nuevo = guardar_archivo_modificado(df, archivo, f"{columnas}_abs")
    return HttpResponseRedirect(reverse('interfaz_procesamiento', args=[id]))


def aplicar_normalizacion(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    ruta = archivo.archivo.path
    df = pd.read_csv(ruta, sep=None, engine='python', on_bad_lines='skip')

    columnas = request.POST.getlist("columnas")

    col_data = df[columnas]
    df[columnas] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
    nuevo = guardar_archivo_modificado(df, archivo, f"{columnas}_norm")
    return HttpResponseRedirect(reverse('interfaz_procesamiento', args=[id]))

import traceback


import seaborn as sns
import matplotlib.pyplot as plt
import io, base64

def generar_matriz_correlacion(df):
    try:
        corr = df.corr(numeric_only=True)
        if corr.empty:
            return None

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Matriz de correlación", fontsize=14)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print("[ERROR] al generar matriz de correlación:", e)
        return None



@login_required
def interfaz_procesamiento(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    ruta = archivo.archivo.path

    # Leer archivo y eliminar filas completamente vacías
    df = pd.read_csv(ruta, sep=None, engine='python', on_bad_lines='skip')
    df.dropna(how='all', inplace=True)

    columnas = df.columns.tolist()
    columnas_numericas = df.select_dtypes(include='number').columns.tolist()
    

    boxplot = generar_boxplot(df)
    tendencia = generar_tendencia(df)
    comparativo_boxplot = None
    correlacion_img = generar_matriz_correlacion(df)

    mensaje = None
    nuevo_archivo = None


    # === Nuevo bloque: genera comparativo automáticamente si el archivo es modificado ===
    if "_" in archivo.nombre:  # Parece un archivo modificado
        try:
            # Obtener el nombre del archivo original
            nombre_original = archivo.nombre.split('_')[0] + ".csv"
            ruta_original = os.path.join(settings.MEDIA_ROOT, 'uploads', nombre_original)

            if os.path.exists(ruta_original):
                df_original = pd.read_csv(ruta_original, sep=None, engine='python', on_bad_lines='skip')
                df_original.dropna(how='all', inplace=True)

                # Generar el boxplot del archivo original
                comparativo_boxplot = generar_boxplot(df_original)
        except Exception as e:
            print("[ERROR] Al generar boxplot del archivo original:", e)


    if request.method == 'POST':
            columnas = request.POST.getlist("columnas")
            operacion = request.POST.get('operacion')
            valor_personalizado = request.POST.get('valor_personalizado')

            df_filtrado = df.copy()
            sufijos = []

            if operacion == "eliminar_negativos":
                for col in columnas:
                    if (df_filtrado[col] < 0).any():
                        df_filtrado = df_filtrado[df_filtrado[col] >= 0]
                sufijos.append("sin_negativos")

            elif operacion == "eliminar_ceros":
                for col in columnas:
                    if (df_filtrado[col] == 0).any():
                        df_filtrado = df_filtrado[df_filtrado[col] != 0]
                sufijos.append("sin_ceros")

            elif operacion == "eliminar_nulos":
                for col in columnas:
                    if df_filtrado[col].isnull().any():
                        df_filtrado = df_filtrado[df_filtrado[col].notnull()]
                sufijos.append("sin_nulos")

            elif operacion == "valor_personalizado" and valor_personalizado:
                try:
                    valor = float(valor_personalizado)
                    for col in columnas:
                        df_filtrado = df_filtrado[df_filtrado[col] != valor]
                    sufijos.append(f"sin_{valor}")
                except ValueError:
                    mensaje = "Valor personalizado inválido."

            elif operacion == "abs":
                df_filtrado[columnas] = df_filtrado[columnas].abs()
                sufijos.append("abs")

            elif operacion == "normalizar":
                for col in columnas:
                    max_val = df_filtrado[col].max()
                    min_val = df_filtrado[col].min()
                    if max_val != min_val:
                        df_filtrado[col] = (df_filtrado[col] - min_val) / (max_val - min_val)
                    else:
                        mensaje = f"No se puede normalizar columna {col}: max y min son iguales."
                sufijos.append("norm")

            # Guardar nuevo archivo solo si no hay errores
            if not mensaje:
                nuevo_nombre = guardar_archivo_modificado(
                    df_filtrado, archivo, '_'.join(sufijos) + '_' + uuid.uuid4().hex[:6]
                )
                nuevo_archivo = ArchivoSubido(nombre=nuevo_nombre, archivo=os.path.join('uploads', nuevo_nombre))
                nuevo_archivo.save()
                mensaje = f"Operación aplicada. Nuevo archivo: {nuevo_archivo.nombre}"
                



    context = {
        'archivo': archivo,
        'columnas': columnas,
        'columnas_numericas': columnas_numericas,
        'boxplot': boxplot,
        'tendencia': tendencia,
        'mensaje': mensaje,
        'nuevo_archivo': nuevo_archivo,
        'comparativo_boxplot': comparativo_boxplot,
        'correlacion_img': correlacion_img,
    }
    return render(request, 'interfaz_procesamiento.html', context)


def generar_boxplot(df):
    try:
        # Filtrar solo columnas numéricas para el boxplot
        df_numerico = df.select_dtypes(include='number')

        if df_numerico.empty:
            return None  # No hay columnas numéricas para graficar

        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df_numerico)
        plt.xticks(rotation=45)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()

        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')

        return graphic
    except Exception as e:
        print(f"Error generando boxplot: {e}")
        return None

def generar_tendencia(df):
    df_numerico = df.select_dtypes(include='number')

    if df_numerico.empty:
        return None

    plt.figure(figsize=(10, 4))
    df_numerico.plot(ax=plt.gca())
    plt.xlabel('Índice')
    plt.ylabel('Valores')
    plt.title('Tendencia')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    imagen_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return imagen_base64

"""
def interfaz_modelado(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    modelo_info = None
    metricas = None

    try:
        df = pd.read_csv(archivo.archivo.path, sep=None, engine='python', on_bad_lines='skip')
        df.dropna(inplace=True)
        
        # Identificar columnas numéricas y categóricas
        columnas_numericas = df.select_dtypes(include='number').columns.tolist()
        columnas_categoricas = df.select_dtypes(exclude='number').columns.tolist()
        
        # Convertir columnas categóricas si existen
        if columnas_categoricas:
            le = LabelEncoder()
            for col in columnas_categoricas:
                df[col] = le.fit_transform(df[col])
            columnas_numericas = df.columns.tolist()  # Ahora todas son numéricas

    except Exception as e:
        messages.error(request, f"Error al cargar archivo: {e}")
        return render(request, "interfaz_modelado.html", {
            "archivo": archivo,
            "columnas": [],
            "modelo_info": None,
            "metricas": None
        })

    if request.method == "POST":
        columna_objetivo = request.POST.get("columna_objetivo")
        columnas_predictoras = request.POST.getlist("columnas_predictoras")
        valor_str = request.POST.get("valor", "").strip()

        # Validaciones
        if not columna_objetivo:
            messages.error(request, "Debes seleccionar una columna objetivo.")
            return render(request, "interfaz_modelado.html", {
                "archivo": archivo,
                "columnas": columnas_numericas,
                "modelo_info": None,
                "metricas": None
            })

        if not columnas_predictoras:
            messages.error(request, "Debes seleccionar al menos una columna predictora.")
            return render(request, "interfaz_modelado.html", {
                "archivo": archivo,
                "columnas": columnas_numericas,
                "modelo_info": None,
                "metricas": None
            })

        if columna_objetivo in columnas_predictoras:
            messages.error(request, "La columna objetivo no puede estar entre las predictoras.")
            return render(request, "interfaz_modelado.html", {
                "archivo": archivo,
                "columnas": columnas_numericas,
                "modelo_info": None,
                "metricas": None
            })

        try:
            # Preparar datos
            X = df[columnas_predictoras]
            y = df[columna_objetivo]
            
            # Entrenar modelo
            modelo = RandomForestRegressor(n_estimators=100, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            modelo.fit(X_train, y_train)
            
            # Calcular métricas
            y_pred = modelo.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metricas = {
                'mae': round(mae, 4),
                'r2': round(r2, 4),
                'n_estimators': modelo.n_estimators,
                'features_importances': sorted(
                    zip(columnas_predictoras, modelo.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True
                )
            }
            
            modelo_info = f"Modelo RandomForest entrenado para predecir '{columna_objetivo}' usando {len(columnas_predictoras)} predictores."
            
            # Hacer predicción si se proporcionan valores
            if valor_str:
                try:
                    nuevo_valor = [float(v.strip()) for v in valor_str.split(",")]
                    if len(nuevo_valor) != len(columnas_predictoras):
                        messages.error(
                            request,
                            f"Debes ingresar {len(columnas_predictoras)} valores separados por coma (uno por cada predictor)."
                        )
                    else:
                        prediccion = modelo.predict([nuevo_valor])[0]
                        messages.success(
                            request,
                            f"Predicción para '{columna_objetivo}': {prediccion:.4f}"
                        )
                except ValueError:
                    messages.error(request, "Los valores de entrada deben ser números separados por comas.")
                
        except Exception as e:
            messages.error(request, f"Error durante el modelado: {str(e)}")

    return render(request, "interfaz_modelado.html", {
        "archivo": archivo,
        "columnas": columnas_numericas,
        "modelo_info": modelo_info,
        "metricas": metricas
    })
"""
from django.shortcuts import render, get_object_or_404
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
import seaborn as sns
#nueva parte
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')
import json
from io import StringIO

def interfaz_modelado(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    modelo_info = None
    metricas = None
    df_predicciones = None
    df_parcial_predicciones = None
    es_clasificacion = False
    scatter_img = None
    displot_img = None
    crosscorrelation_img = None
    feature_importances_img = None

    # Variables para estadísticas
    total_filas = 0
    conocidos_filas = 0
    predichos_filas = 0
    error_promedio = 0
    accuracy_parcial = 0
    

    # Valores por defecto de los parámetros
    parametros = {
        'test_size': 0.2,
        'random_state_split': 42,
        'n_estimators': 100,
        'random_state_model': 42,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    
    # Valores para mostrar en la plantilla
    test_size_percent = 20
    train_size_percent = 80

    try:
        df = pd.read_csv(archivo.archivo.path, sep=None, engine='python', on_bad_lines='skip')
        df.dropna(inplace=True)
        
        # Identificar columnas numéricas y categóricas
        columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
        columnas_categoricas = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Convertir columnas categóricas si existen
        if columnas_categoricas:
            for col in columnas_categoricas:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()

    except Exception as e:
        messages.error(request, f"Error al cargar archivo: {e}")
        return render(request, "interfaz_modelado.html", {
            "archivo": archivo,
            "columnas": [],
            "modelo_info": None,
            "metricas": None,
            "df_predicciones": None,
            "df_parcial_predicciones": None,
            "total_filas": total_filas,
            "conocidos_filas": conocidos_filas,
            "predichos_filas": predichos_filas,
            "error_promedio": error_promedio,
            "accuracy_parcial": accuracy_parcial,
            "parametros": parametros,
            "test_size_percent": test_size_percent,
            "train_size_percent": train_size_percent
        })

    if request.method == "POST":
        columna_objetivo = request.POST.get("columna_objetivo")
        columnas_predictoras = request.POST.getlist("columnas_predictoras")
        valor_str = request.POST.get("valor", "").strip()
        predecir_columna = request.POST.get("predecir_columna") == "on"
        predecir_parcial = request.POST.get("predecir_parcial") == "on"
        porcentaje_datos = float(request.POST.get("porcentaje_datos", 50))
        numeric_df = df.select_dtypes(include=[np.number])

        # Obtener parámetros ajustables del usuario
        try:
            # test_size como porcentaje
            test_size_percent = float(request.POST.get("test_size", 20))
            parametros['test_size'] = test_size_percent / 100.0
            train_size_percent = 100 - test_size_percent
            
            # Parámetros principales
            parametros['random_state_split'] = int(request.POST.get("random_state_split", 42))
            parametros['n_estimators'] = int(request.POST.get("n_estimators", 100))
            parametros['random_state_model'] = int(request.POST.get("random_state_model", 42))
            
            # Parámetros avanzados
            max_depth_str = request.POST.get("max_depth", "")
            parametros['max_depth'] = int(max_depth_str) if max_depth_str and max_depth_str != "None" else None
            
            parametros['min_samples_split'] = int(request.POST.get("min_samples_split", 2))
            parametros['min_samples_leaf'] = int(request.POST.get("min_samples_leaf", 1))
            
        except (ValueError, TypeError):
            messages.warning(request, "Parámetros inválidos, usando valores por defecto.")
            parametros = {
                'test_size': 0.2,
                'random_state_split': 42,
                'n_estimators': 100,
                'random_state_model': 42,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
            test_size_percent = 20
            train_size_percent = 80
        
    

        # Validaciones
        if not columna_objetivo or columna_objetivo not in df.columns:
            messages.error(request, "Debes seleccionar una columna objetivo válida.")
            return render(request, "interfaz_modelado.html", {
                "archivo": archivo,
                "columnas": columnas_numericas,
                "modelo_info": None,
                "metricas": None,
                "df_predicciones": None,
                "df_parcial_predicciones": None,
                "total_filas": total_filas,
                "conocidos_filas": conocidos_filas,
                "predichos_filas": predichos_filas,
                "error_promedio": error_promedio,
                "accuracy_parcial": accuracy_parcial,
                "parametros": parametros,
                "test_size_percent": test_size_percent,
                "train_size_percent": train_size_percent
            })

        if not columnas_predictoras or not all(col in df.columns for col in columnas_predictoras):
            messages.error(request, "Debes seleccionar columnas predictoras válidas.")
            return render(request, "interfaz_modelado.html", {
                "archivo": archivo,
                "columnas": columnas_numericas,
                "modelo_info": None,
                "metricas": None,
                "df_predicciones": None,
                "df_parcial_predicciones": None,
                "total_filas": total_filas,
                "conocidos_filas": conocidos_filas,
                "predichos_filas": predichos_filas,
                "error_promedio": error_promedio,
                "accuracy_parcial": accuracy_parcial,
                "parametros": parametros,
                "test_size_percent": test_size_percent,
                "train_size_percent": train_size_percent
            })

        if columna_objetivo in columnas_predictoras:
            messages.error(request, "La columna objetivo no puede estar entre las predictoras.")
            return render(request, "interfaz_modelado.html", {
                "archivo": archivo,
                "columnas": columnas_numericas,
                "modelo_info": None,
                "metricas": None,
                "df_predicciones": None,
                "df_parcial_predicciones": None,
                "total_filas": total_filas,
                "conocidos_filas": conocidos_filas,
                "predichos_filas": predichos_filas,
                "error_promedio": error_promedio,
                "accuracy_parcial": accuracy_parcial,
                "parametros": parametros,
                "test_size_percent": test_size_percent,
                "train_size_percent": train_size_percent
            })

        try:
            # Preparar datos
            X = df[columnas_predictoras].apply(pd.to_numeric, errors='coerce')
            y = df[columna_objetivo].apply(pd.to_numeric, errors='coerce')
            
            # Eliminar filas con NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                messages.error(request, "No hay datos válidos después de la limpieza.")
                return render(request, "interfaz_modelado.html", {
                    "archivo": archivo,
                    "columnas": columnas_numericas,
                    "modelo_info": None,
                    "metricas": None,
                    "df_predicciones": None,
                    "df_parcial_predicciones": None,
                    "total_filas": total_filas,
                    "conocidos_filas": conocidos_filas,
                    "predichos_filas": predichos_filas,
                    "error_promedio": error_promedio,
                    "accuracy_parcial": accuracy_parcial,
                    "parametros": parametros,
                    "test_size_percent": test_size_percent,
                    "train_size_percent": train_size_percent
                })
            
            # Determinar si es problema de clasificación o regresión
            if y.nunique() <= 10:
                es_clasificacion = True
                modelo = RandomForestClassifier(
                    n_estimators=parametros['n_estimators'],
                    random_state=parametros['random_state_model'],
                    max_depth=parametros['max_depth'],
                    min_samples_split=parametros['min_samples_split'],
                    min_samples_leaf=parametros['min_samples_leaf']
                )
            else:
                modelo = RandomForestRegressor(
                    n_estimators=parametros['n_estimators'],
                    random_state=parametros['random_state_model'],
                    max_depth=parametros['max_depth'],
                    min_samples_split=parametros['min_samples_split'],
                    min_samples_leaf=parametros['min_samples_leaf']
                )
            
            # Entrenar modelo
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=parametros['test_size'],
                random_state=parametros['random_state_split']
            )
            
            modelo.fit(X_train, y_train)
            
            # Calcular métricas
            y_pred = modelo.predict(X_test)
            

            # Crear gráfico scatter solo si es regresión
            if not es_clasificacion:
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))

                    # Eje X como índice temporal
                    x_axis = range(len(y_test))

                    # Puntos de valores reales
                    ax.scatter(x_axis, y_test.values, label="Datos reales", color="blue", alpha=0.6, s=30)

                    # Puntos de predicción
                    ax.scatter(x_axis, y_pred, label="Predicción", color="green", alpha=0.6, s=30)

                    ax.set_xlabel("Tiempo (índice de muestra)")
                    ax.set_ylabel(columna_objetivo)
                    ax.set_title(f"Comparación de valores reales vs predichos ({columna_objetivo})")
                    ax.legend()

                    # Guardar como imagen base64
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format="png")
                    plt.close(fig)
                    buf.seek(0)
                    scatter_img = base64.b64encode(buf.getvalue()).decode("utf-8")

                    # Generar gráfico displot (real vs predicho)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(y_test, color="blue", kde=True, label="Real", stat="density", ax=ax, alpha=0.4)
                    sns.histplot(y_pred, color="green", kde=True, label="Predicho", stat="density", ax=ax, alpha=0.4)
                    ax.set_title("Distribución Real vs Predicho")
                    ax.set_xlabel("Valores")
                    ax.set_ylabel("Densidad")
                    ax.legend()

                    # Convertir a base64
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    displot_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    plt.close(fig)

                    # Gráfico de correlación cruzada (50% de los datos)
                    df_comparacion = pd.DataFrame({
                        "Índice": range(len(y_test)),
                        "y_real": y_test,
                        "y_pred": y_pred
                    })
                    #importancia de variables
                    


                    # Tomar el 50% aleatorio
                    df_sample = df_comparacion.sample(frac=0.5, random_state=42)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(x="Índice", y="y_real", data=df_sample, ax=ax, color="orange", label="Real")
                    sns.scatterplot(x="Índice", y="y_pred", data=df_sample, ax=ax, color="blue", label="Predicho")

                    ax.set_title(f"Comparación Real vs Predicho ({columna_objetivo}) - Muestra 50%")
                    ax.set_xlabel("Índice (muestra)")
                    ax.set_ylabel("Valor")
                    ax.legend()

                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    crosscorrelation_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    plt.close(fig)

                except Exception as e:
                    print("Error al generar gráfico:", e)

            if es_clasificacion:
                accuracy = accuracy_score(y_test, y_pred)
                metricas = {
                    'accuracy': round(accuracy, 4),
                    'classification_report': classification_report(y_test, y_pred),
                    'n_estimators': modelo.n_estimators,
                    'features_importances': sorted(
                        zip(columnas_predictoras, modelo.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True
                    ),
                    'tipo': 'clasificación',
                    'test_size': parametros['test_size'],
                    'test_size_percent': test_size_percent,
                    'random_state_split': parametros['random_state_split'],
                    'random_state_model': parametros['random_state_model'],
                    'max_depth': parametros['max_depth'],
                    'min_samples_split': parametros['min_samples_split'],
                    'min_samples_leaf': parametros['min_samples_leaf']
                }
            else:
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metricas = {
                    'mae': round(mae, 4),
                    'r2': round(r2, 4),
                    'n_estimators': modelo.n_estimators,
                    'features_importances': sorted(
                        zip(columnas_predictoras, modelo.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True
                    ),
                    'tipo': 'regresión',
                    'test_size': parametros['test_size'],
                    'test_size_percent': test_size_percent,
                    'random_state_split': parametros['random_state_split'],
                    'random_state_model': parametros['random_state_model'],
                    'max_depth': parametros['max_depth'],
                    'min_samples_split': parametros['min_samples_split'],
                    'min_samples_leaf': parametros['min_samples_leaf']
                }
            
            modelo_info = f"Modelo RandomForest entrenado para predecir '{columna_objetivo}' usando {len(columnas_predictoras)} predictores."
            
            # Hacer predicción si se proporcionan valores
            if valor_str:
                try:
                    nuevo_valor = [float(v.strip()) for v in valor_str.split(",")]
                    if len(nuevo_valor) != len(columnas_predictoras):
                        messages.error(
                            request,
                            f"Debes ingresar {len(columnas_predictoras)} valores separados por coma (uno por cada predictor)."
                        )
                    else:
                        prediccion = modelo.predict([nuevo_valor])[0]
                        messages.success(
                            request,
                            f"Predicción para '{columna_objetivo}': {prediccion:.4f}"
                        )
                except ValueError:
                    messages.error(request, "Los valores de entrada deben ser números separados por comas.")
            
            # Predecir columna completa
            if predecir_columna:
                predicciones = modelo.predict(X)
                df_predicciones = df.loc[X.index].copy()
                df_predicciones[f'Predicción_{columna_objetivo}'] = predicciones
                df_predicciones['Error'] = np.abs(y.values - predicciones) if not es_clasificacion else (y.values != predicciones).astype(int)
                messages.success(request, f"Se han generado predicciones para {len(df_predicciones)} filas.")
            
            # Predicción parcial de datos
            if predecir_parcial:
                n_filas = len(X)
                n_entrenamiento = int(n_filas * (porcentaje_datos / 100))
                indices = X.index.tolist()
                random.shuffle(indices)
                indices_con_target = indices[:n_entrenamiento]
                indices_sin_target = indices[n_entrenamiento:]
                
                if not indices_sin_target:
                    messages.warning(request, "No hay datos para predecir con el porcentaje seleccionado.")
                else:
                    # Entrenar modelo solo con datos conocidos
                    X_conocido = X.loc[indices_con_target]
                    y_conocido = y.loc[indices_con_target]
                    
                    if es_clasificacion:
                        modelo_parcial = RandomForestClassifier(
                            n_estimators=parametros['n_estimators'],
                            random_state=parametros['random_state_model'],
                            max_depth=parametros['max_depth'],
                            min_samples_split=parametros['min_samples_split'],
                            min_samples_leaf=parametros['min_samples_leaf']
                        )
                    else:
                        modelo_parcial = RandomForestRegressor(
                            n_estimators=parametros['n_estimators'],
                            random_state=parametros['random_state_model'],
                            max_depth=parametros['max_depth'],
                            min_samples_split=parametros['min_samples_split'],
                            min_samples_leaf=parametros['min_samples_leaf']
                        )
                    
                    modelo_parcial.fit(X_conocido, y_conocido)
                    predicciones_todas = modelo_parcial.predict(X)
                    
                    df_parcial_predicciones = df.loc[X.index].copy()
                    df_parcial_predicciones[f'Predicción_{columna_objetivo}'] = predicciones_todas
                    
                    if es_clasificacion:
                        df_parcial_predicciones['Error'] = (y.values != predicciones_todas).astype(int)
                    else:
                        df_parcial_predicciones['Error'] = np.abs(y.values - predicciones_todas)
                    
                    df_parcial_predicciones['Tipo'] = 'Predicho'
                    df_parcial_predicciones.loc[indices_con_target, 'Tipo'] = 'Conocido (Entrenamiento)'
                    
                    total_filas = len(df_parcial_predicciones)
                    conocidos_filas = len(indices_con_target)
                    predichos_filas = len(indices_sin_target)
                    
                    if conocidos_filas > 0:
                        datos_entrenamiento = df_parcial_predicciones.loc[indices_con_target]
                        if es_clasificacion:
                            correctos = len(datos_entrenamiento[datos_entrenamiento['Error'] == 0])
                            accuracy_parcial = round((correctos / conocidos_filas) * 100, 2)
                        else:
                            error_promedio = round(datos_entrenamiento['Error'].mean(), 4)
                    
                    messages.success(request, f"Predicción parcial completada. {conocidos_filas} filas para entrenamiento, {predichos_filas} filas predichas.")
                
        except Exception as e:
            messages.error(request, f"Error durante el modelado: {str(e)}")
            import traceback
            traceback.print_exc()

    return render(request, "interfaz_modelado.html", {
        "archivo": archivo,
        "columnas": columnas_numericas,
        "modelo_info": modelo_info,
        "metricas": metricas,
        "df_predicciones": df_predicciones,
        "df_parcial_predicciones": df_parcial_predicciones,
        "es_clasificacion": es_clasificacion,
        "total_filas": total_filas,
        "conocidos_filas": conocidos_filas,
        "predichos_filas": predichos_filas,
        "error_promedio": error_promedio,
        "accuracy_parcial": accuracy_parcial,
        "parametros": parametros,
        "test_size_percent": test_size_percent,
        "train_size_percent": train_size_percent,
        "scatter_img": scatter_img,
        "displot_img": displot_img,
        "crosscorrelation_img": crosscorrelation_img,

    })

def descargar_predicciones(request, id):
    if request.method == "POST":
        datos_predicciones = request.POST.get("datos_predicciones")
        archivo = get_object_or_404(ArchivoSubido, id=id)
        
        try:
            df = pd.read_json(StringIO(datos_predicciones))
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="predicciones_{archivo.nombre}.csv"'
            df.to_csv(response, index=False)
            return response
            
        except Exception as e:
            messages.error(request, f"Error al generar el archivo: {str(e)}")
            return redirect('interfaz_modelado', id=id)

from django.shortcuts import render, get_object_or_404
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
import seaborn as sns
#nueva parte
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')
#cross validation y learning curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
#Grid mejores valores para las variables
from sklearn.model_selection import GridSearchCV

def prueba_modelado_randomforest(request, id):
    archivo = get_object_or_404(ArchivoSubido, id=id)
    modelo_info = None
    metricas = None
    df_predicciones = None
    df_parcial_predicciones = None
    es_clasificacion = False
    scatter_img = None
    displot_img = None
    crosscorrelation_img = None
    learning_curve_img = None
    feature_importances_img = None

    # Variables para estadísticas
    total_filas = 0
    conocidos_filas = 0
    predichos_filas = 0
    error_promedio = 0
    accuracy_parcial = 0
    
    #grill de parametros
    # Grilla de parámetros
    

    # Valores por defecto de los parámetros
    parametros = {
        'test_size': 0.2,
        'random_state_split': 42,
        'n_estimators': 100,
        'random_state_model': 42,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    
    # Valores para mostrar en la plantilla
    test_size_percent = 20
    train_size_percent = 80

    try:
        df = pd.read_csv(archivo.archivo.path, sep=None, engine='python', on_bad_lines='skip')
        df.dropna(inplace=True)
        
        # Identificar columnas numéricas y categóricas
        columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()
        columnas_categoricas = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Convertir columnas categóricas si existen
        if columnas_categoricas:
            for col in columnas_categoricas:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            columnas_numericas = df.select_dtypes(include=['number']).columns.tolist()

    except Exception as e:
        messages.error(request, f"Error al cargar archivo: {e}")
        return render(request, "prueba_modelado_randomforest.html", {
            "archivo": archivo,
            "columnas": [],
            "modelo_info": None,
            "metricas": None,
            "df_predicciones": None,
            "df_parcial_predicciones": None,
            "total_filas": total_filas,
            "conocidos_filas": conocidos_filas,
            "predichos_filas": predichos_filas,
            "error_promedio": error_promedio,
            "accuracy_parcial": accuracy_parcial,
            "parametros": parametros,
            "test_size_percent": test_size_percent,
            "train_size_percent": train_size_percent
        })

    if request.method == "POST":
        columna_objetivo = request.POST.get("columna_objetivo")
        columnas_predictoras = request.POST.getlist("columnas_predictoras")
        valor_str = request.POST.get("valor", "").strip()
        predecir_columna = request.POST.get("predecir_columna") == "on"
        predecir_parcial = request.POST.get("predecir_parcial") == "on"
        porcentaje_datos = float(request.POST.get("porcentaje_datos", 50))
        numeric_df = df.select_dtypes(include=[np.number])

        # Obtener parámetros ajustables del usuario
        try:
            # test_size como porcentaje
            test_size_percent = float(request.POST.get("test_size", 20))
            parametros['test_size'] = test_size_percent / 100.0
            train_size_percent = 100 - test_size_percent
            
            # Parámetros principales
            parametros['random_state_split'] = int(request.POST.get("random_state_split", 42))
            parametros['n_estimators'] = int(request.POST.get("n_estimators", 100))
            parametros['random_state_model'] = int(request.POST.get("random_state_model", 42))
            
            # Parámetros avanzados
            max_depth_str = request.POST.get("max_depth", "")
            parametros['max_depth'] = int(max_depth_str) if max_depth_str and max_depth_str != "None" else None
            
            parametros['min_samples_split'] = int(request.POST.get("min_samples_split", 2))
            parametros['min_samples_leaf'] = int(request.POST.get("min_samples_leaf", 1))
            
        except (ValueError, TypeError):
            messages.warning(request, "Parámetros inválidos, usando valores por defecto.")
            parametros = {
                'test_size': 0.2,
                'random_state_split': 42,
                'n_estimators': 100,
                'random_state_model': 42,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
            test_size_percent = 20
            train_size_percent = 80
        
    

        # Validaciones
        if not columna_objetivo or columna_objetivo not in df.columns:
            messages.error(request, "Debes seleccionar una columna objetivo válida.")
            return render(request, "prueba_modelado_randomforest.html", {
                "archivo": archivo,
                "columnas": columnas_numericas,
                "modelo_info": None,
                "metricas": None,
                "df_predicciones": None,
                "df_parcial_predicciones": None,
                "total_filas": total_filas,
                "conocidos_filas": conocidos_filas,
                "predichos_filas": predichos_filas,
                "error_promedio": error_promedio,
                "accuracy_parcial": accuracy_parcial,
                "parametros": parametros,
                "test_size_percent": test_size_percent,
                "train_size_percent": train_size_percent
            })

        if not columnas_predictoras or not all(col in df.columns for col in columnas_predictoras):
            messages.error(request, "Debes seleccionar columnas predictoras válidas.")
            return render(request, "prueba_modelado_randomforest.html", {
                "archivo": archivo,
                "columnas": columnas_numericas,
                "modelo_info": None,
                "metricas": None,
                "df_predicciones": None,
                "df_parcial_predicciones": None,
                "total_filas": total_filas,
                "conocidos_filas": conocidos_filas,
                "predichos_filas": predichos_filas,
                "error_promedio": error_promedio,
                "accuracy_parcial": accuracy_parcial,
                "parametros": parametros,
                "test_size_percent": test_size_percent,
                "train_size_percent": train_size_percent
            })

        if columna_objetivo in columnas_predictoras:
            messages.error(request, "La columna objetivo no puede estar entre las predictoras.")
            return render(request, "prueba_modelado_randomforest.html", {
                "archivo": archivo,
                "columnas": columnas_numericas,
                "modelo_info": None,
                "metricas": None,
                "df_predicciones": None,
                "df_parcial_predicciones": None,
                "total_filas": total_filas,
                "conocidos_filas": conocidos_filas,
                "predichos_filas": predichos_filas,
                "error_promedio": error_promedio,
                "accuracy_parcial": accuracy_parcial,
                "parametros": parametros,
                "test_size_percent": test_size_percent,
                "train_size_percent": train_size_percent
            })

        try:
            # Preparar datos
            X = df[columnas_predictoras].apply(pd.to_numeric, errors='coerce')
            y = df[columna_objetivo].apply(pd.to_numeric, errors='coerce')
            
            # Eliminar filas con NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            


            if len(X) == 0:
                messages.error(request, "No hay datos válidos después de la limpieza.")
                return render(request, "prueba_modelado_randomforest.html", {
                    "archivo": archivo,
                    "columnas": columnas_numericas,
                    "modelo_info": None,
                    "metricas": None,
                    "df_predicciones": None,
                    "df_parcial_predicciones": None,
                    "total_filas": total_filas,
                    "conocidos_filas": conocidos_filas,
                    "predichos_filas": predichos_filas,
                    "error_promedio": error_promedio,
                    "accuracy_parcial": accuracy_parcial,
                    "parametros": parametros,
                    "test_size_percent": test_size_percent,
                    "train_size_percent": train_size_percent
                })
            

            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }

            # Determinar si es problema de clasificación o regresión
            if y.nunique() <= 10:
                es_clasificacion = True
                modelo = RandomForestClassifier(random_state=parametros['random_state_model'])
                scoring = "accuracy"
            else:
                modelo = RandomForestRegressor(random_state=parametros['random_state_model'])
                scoring = "r2"
            
            grid_search = GridSearchCV(
                modelo,
                param_grid,
                cv=5,
                scoring=scoring,
                n_jobs=-1
            )

            grid_search.fit(X, y)

            # Extraer mejores parámetros y modelo
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            modelo = grid_search.best_estimator_

            # Entrenar modelo
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=parametros['test_size'],
                random_state=parametros['random_state_split']
            )
            
            modelo.fit(X_train, y_train)
            
            # learning curve
            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    modelo, X, y, cv=5, scoring="r2",
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    n_jobs=-1
                )

                train_scores_mean = np.mean(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(train_sizes, train_scores_mean, "o-", color="blue", label="Entrenamiento")
                ax.plot(train_sizes, test_scores_mean, "o-", color="green", label="Validación")
                ax.set_title("Curva de Aprendizaje (Cross-validation)")
                ax.set_xlabel("Tamaño del conjunto de entrenamiento")
                ax.set_ylabel("Score (R²)")
                ax.legend()

                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                learning_curve_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                plt.close(fig)

                #importancia de variables
                importances = modelo.feature_importances_
                features = columnas_predictoras
                indices = np.argsort(importances)[::-1]

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis", ax=ax)

                ax.set_title("Importancia de las características")
                ax.set_xlabel("Importancia")
                ax.set_ylabel("Características")

                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                feature_importances_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                plt.close(fig)
            

            except Exception as e:
                print("Error al generar learning curve:", e)

            # Cross-validation (5 particiones por defecto)
            cv_scores = cross_val_score(modelo, X, y, cv=5, scoring="r2" if not es_clasificacion else "accuracy")
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Calcular métricas
            y_pred = modelo.predict(X_test)
            

            # Crear gráfico scatter solo si es regresión
            if not es_clasificacion:
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))

                    # Eje X como índice temporal
                    x_axis = range(len(y_test))

                    # Puntos de valores reales
                    ax.scatter(x_axis, y_test.values, label="Datos reales", color="blue", alpha=0.6, s=30)

                    # Puntos de predicción
                    ax.scatter(x_axis, y_pred, label="Predicción", color="green", alpha=0.6, s=30)

                    ax.set_xlabel("Tiempo (índice de muestra)")
                    ax.set_ylabel(columna_objetivo)
                    ax.set_title(f"Comparación de valores reales vs predichos ({columna_objetivo})")
                    ax.legend()

                    # Guardar como imagen base64
                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format="png")
                    plt.close(fig)
                    buf.seek(0)
                    scatter_img = base64.b64encode(buf.getvalue()).decode("utf-8")

                    # Generar gráfico displot (real vs predicho)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(y_test, color="blue", kde=True, label="Real", stat="density", ax=ax, alpha=0.4)
                    sns.histplot(y_pred, color="green", kde=True, label="Predicho", stat="density", ax=ax, alpha=0.4)
                    ax.set_title("Distribución Real vs Predicho")
                    ax.set_xlabel("Valores")
                    ax.set_ylabel("Densidad")
                    ax.legend()

                    # Convertir a base64
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    displot_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    plt.close(fig)

                    # Gráfico de correlación cruzada (50% de los datos)
                    df_comparacion = pd.DataFrame({
                        "Índice": range(len(y_test)),
                        "y_real": y_test,
                        "y_pred": y_pred
                    })

                    # Tomar el 50% aleatorio
                    df_sample = df_comparacion.sample(frac=0.5, random_state=42)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(x="Índice", y="y_real", data=df_sample, ax=ax, color="orange", label="Real")
                    sns.scatterplot(x="Índice", y="y_pred", data=df_sample, ax=ax, color="blue", label="Predicho")

                    ax.set_title(f"Comparación Real vs Predicho ({columna_objetivo}) - Muestra 50%")
                    ax.set_xlabel("Índice (muestra)")
                    ax.set_ylabel("Valor")
                    ax.legend()

                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", bbox_inches="tight")
                    buffer.seek(0)
                    crosscorrelation_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    plt.close(fig)
                
                




                except Exception as e:
                    print("Error al generar gráfico:", e)

            if es_clasificacion:
                accuracy = accuracy_score(y_test, y_pred)
                metricas = {
                    'accuracy': round(accuracy, 4),
                    'cv_mean': round(cv_mean, 4),
                    'cv_std': round(cv_std, 4),
                    'best_params': best_params,
                    'best_score': round(best_score, 4),
                    'classification_report': classification_report(y_test, y_pred),
                    'n_estimators': modelo.n_estimators,
                    'features_importances': sorted(
                        zip(columnas_predictoras, modelo.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True
                    ),
                    'tipo': 'clasificación',
                    'test_size': parametros['test_size'],
                    'test_size_percent': test_size_percent,
                    'random_state_split': parametros['random_state_split'],
                    'random_state_model': parametros['random_state_model'],
                    'max_depth': parametros['max_depth'],
                    'min_samples_split': parametros['min_samples_split'],
                    'min_samples_leaf': parametros['min_samples_leaf']
                }
            else:
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                metricas = {
                    'mae': round(mae, 4),
                    'r2': round(r2, 4),
                    'best_params': best_params,
                    'best_score': round(best_score, 4),
                    'cv_mean': round(cv_mean, 4),
                    'cv_std': round(cv_std, 4),
                    'n_estimators': modelo.n_estimators,
                    'features_importances': sorted(
                        zip(columnas_predictoras, modelo.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True
                    ),
                    'tipo': 'regresión',
                    'test_size': parametros['test_size'],
                    'test_size_percent': test_size_percent,
                    'random_state_split': parametros['random_state_split'],
                    'random_state_model': parametros['random_state_model'],
                    'max_depth': parametros['max_depth'],
                    'min_samples_split': parametros['min_samples_split'],
                    'min_samples_leaf': parametros['min_samples_leaf']
                }
            
            modelo_info = f"Modelo RandomForest entrenado para predecir '{columna_objetivo}' usando {len(columnas_predictoras)} predictores."
            
            # Hacer predicción si se proporcionan valores
            if valor_str:
                try:
                    nuevo_valor = [float(v.strip()) for v in valor_str.split(",")]
                    if len(nuevo_valor) != len(columnas_predictoras):
                        messages.error(
                            request,
                            f"Debes ingresar {len(columnas_predictoras)} valores separados por coma (uno por cada predictor)."
                        )
                    else:
                        prediccion = modelo.predict([nuevo_valor])[0]
                        messages.success(
                            request,
                            f"Predicción para '{columna_objetivo}': {prediccion:.4f}"
                        )
                except ValueError:
                    messages.error(request, "Los valores de entrada deben ser números separados por comas.")
            
            # Predecir columna completa
            if predecir_columna:
                predicciones = modelo.predict(X)
                df_predicciones = df.loc[X.index].copy()
                df_predicciones[f'Predicción_{columna_objetivo}'] = predicciones
                df_predicciones['Error'] = np.abs(y.values - predicciones) if not es_clasificacion else (y.values != predicciones).astype(int)
                messages.success(request, f"Se han generado predicciones para {len(df_predicciones)} filas.")
            
            # Predicción parcial de datos
            if predecir_parcial:
                n_filas = len(X)
                n_entrenamiento = int(n_filas * (porcentaje_datos / 100))
                indices = X.index.tolist()
                random.shuffle(indices)
                indices_con_target = indices[:n_entrenamiento]
                indices_sin_target = indices[n_entrenamiento:]
                
                if not indices_sin_target:
                    messages.warning(request, "No hay datos para predecir con el porcentaje seleccionado.")
                else:
                    # Entrenar modelo solo con datos conocidos
                    X_conocido = X.loc[indices_con_target]
                    y_conocido = y.loc[indices_con_target]
                    
                    if es_clasificacion:
                        modelo_parcial = RandomForestClassifier(
                            n_estimators=parametros['n_estimators'],
                            random_state=parametros['random_state_model'],
                            max_depth=parametros['max_depth'],
                            min_samples_split=parametros['min_samples_split'],
                            min_samples_leaf=parametros['min_samples_leaf']
                        )
                    else:
                        modelo_parcial = RandomForestRegressor(
                            n_estimators=parametros['n_estimators'],
                            random_state=parametros['random_state_model'],
                            max_depth=parametros['max_depth'],
                            min_samples_split=parametros['min_samples_split'],
                            min_samples_leaf=parametros['min_samples_leaf']
                        )
                    
                    modelo_parcial.fit(X_conocido, y_conocido)
                    predicciones_todas = modelo_parcial.predict(X)
                    
                    df_parcial_predicciones = df.loc[X.index].copy()
                    df_parcial_predicciones[f'Predicción_{columna_objetivo}'] = predicciones_todas
                    
                    if es_clasificacion:
                        df_parcial_predicciones['Error'] = (y.values != predicciones_todas).astype(int)
                    else:
                        df_parcial_predicciones['Error'] = np.abs(y.values - predicciones_todas)
                    
                    df_parcial_predicciones['Tipo'] = 'Predicho'
                    df_parcial_predicciones.loc[indices_con_target, 'Tipo'] = 'Conocido (Entrenamiento)'
                    
                    total_filas = len(df_parcial_predicciones)
                    conocidos_filas = len(indices_con_target)
                    predichos_filas = len(indices_sin_target)
                    
                    if conocidos_filas > 0:
                        datos_entrenamiento = df_parcial_predicciones.loc[indices_con_target]
                        if es_clasificacion:
                            correctos = len(datos_entrenamiento[datos_entrenamiento['Error'] == 0])
                            accuracy_parcial = round((correctos / conocidos_filas) * 100, 2)
                        else:
                            error_promedio = round(datos_entrenamiento['Error'].mean(), 4)
                    
                    messages.success(request, f"Predicción parcial completada. {conocidos_filas} filas para entrenamiento, {predichos_filas} filas predichas.")
                
        except Exception as e:
            messages.error(request, f"Error durante el modelado: {str(e)}")
            import traceback
            traceback.print_exc()

    return render(request, "prueba_modelado_randomforest.html", {
        "archivo": archivo,
        "columnas": columnas_numericas,
        "modelo_info": modelo_info,
        "metricas": metricas,
        "df_predicciones": df_predicciones,
        "df_parcial_predicciones": df_parcial_predicciones,
        "es_clasificacion": es_clasificacion,
        "total_filas": total_filas,
        "conocidos_filas": conocidos_filas,
        "predichos_filas": predichos_filas,
        "error_promedio": error_promedio,
        "accuracy_parcial": accuracy_parcial,
        "parametros": parametros,
        "test_size_percent": test_size_percent,
        "train_size_percent": train_size_percent,
        "scatter_img": scatter_img,
        "displot_img": displot_img,
        "crosscorrelation_img": crosscorrelation_img,
        "learning_curve_img": learning_curve_img,
        "feature_importances_img": feature_importances_img,
    })


def eliminar_archivo(request, archivo_id):
    archivo = get_object_or_404(ArchivoSubido, pk=archivo_id)
    archivo_path = os.path.join(settings.MEDIA_ROOT, archivo.archivo.name)

    if os.path.exists(archivo_path):
        os.remove(archivo_path)

    archivo.delete()
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    messages.success(request, "Archivo borrado exitosamente")
    return redirect('interfaz')


def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect('interfaz')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})