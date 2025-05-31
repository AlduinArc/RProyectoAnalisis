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
    elif extension in ['.jpg', '.jpeg', '.png', '.gif']:
        try:
            # Redimensionar imagen para previsualización (opcional)
            with Image.open(filepath) as img:
                img.thumbnail((800, 800))  # Ajusta el tamaño máximo
                preview_path = os.path.join(settings.MEDIA_ROOT, f"preview_{archivo.nombre}")
                img.save(preview_path)
            
            contexto = {
                'tipo': 'imagen',
                'imagen_url': archivo.archivo.url,
                'preview_url': f"{settings.MEDIA_URL}preview_{archivo.nombre}"
            }
        except Exception as e:
            contexto = {
                'tipo': 'imagen',
                'error': f"No se pudo procesar la imagen: {str(e)}"
            }

    # Caso 3: PDF
    elif extension == '.pdf':
        contexto = {
            'tipo': 'pdf',
            'pdf_url': archivo.archivo.url
        }

    # Caso 4: Otros tipos (descarga directa)
    else:
        return FileResponse(open(filepath, 'rb'), as_attachment=True)

    # Renderizar según el tipo de archivo
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'partials/ver_contenido.html', contexto)
    return render(request, 'ver_archivo.html', contexto)

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