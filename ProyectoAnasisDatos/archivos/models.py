import os
from django.db import models

class ArchivoSubido(models.Model):
    nombre = models.CharField(max_length=100, blank=True)  # Permite que esté vacío inicialmente
    archivo = models.FileField(upload_to='uploads/')
    descripcion = models.CharField(max_length=255)
    fecha_subida = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # Si no tiene nombre (primera vez que se guarda), lo extrae del archivo
        if not self.nombre:
            self.nombre = os.path.basename(self.archivo.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.nombre