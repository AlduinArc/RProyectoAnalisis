import os
import django
from django.conf import settings

# Configura Django (para que el script pueda acceder a los modelos)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProyectoAnasisDatos.settings')  # Cambia "tu_proyecto"
django.setup()

from archivos.models import ArchivoSubido  # Cambia "tu_app"

def actualizar_nombres():
    for archivo in ArchivoSubido.objects.all():
        nombre_real = os.path.basename(archivo.archivo.name)
        if archivo.nombre != nombre_real:
            archivo.nombre = nombre_real
            archivo.save()
            print(f"Actualizado: {nombre_real}")
        else:
            print(f"Ya estaba correcto: {nombre_real}")

if __name__ == "__main__":
    actualizar_nombres()
    print("¡Todos los nombres han sido actualizados!")