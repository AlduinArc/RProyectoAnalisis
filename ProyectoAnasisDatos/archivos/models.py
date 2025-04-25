from django.db import models

# Create your models here.
class ArchivoSubido(models.Model):
    nombre = models.CharField(max_length=100) #ultimo agregado, para mostrar se quita
    archivo = models.FileField(upload_to='uploads/')
    descripcion = models.CharField(max_length=255)
    fecha_subida = models.DateTimeField(auto_now_add=True) #ultimo agregado, para mostrar se quita

    def __str__(self):
        return self.nombre