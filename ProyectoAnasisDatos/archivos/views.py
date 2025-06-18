# Standard Library
import os
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

# Django Core
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.http import JsonResponse, FileResponse, HttpResponse
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
            df_analizar = df.apply(pd.to_numeric, errors='coerce')
            descripcion_html = df_analizar.describe(include='all').fillna('').to_html(
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

            columnas = df.columns.tolist()

            contexto = {
                'archivo': archivo,
                'columnas': columnas
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
    # Aca una vista temporal para mostrar y opinar del cambio
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

    return render(request, 'temp_ver_analisis_grafico.html', context)

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