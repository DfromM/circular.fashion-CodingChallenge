from django.db import models


class ImageRotations(models.Model):
    img = models.ImageField('label', upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "image"
