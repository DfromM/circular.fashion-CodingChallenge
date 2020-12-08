from rest_framework import serializers

from .models import ImageRotations


class ImageRotationsSerializers(serializers.ModelSerializer):
    class Meta:
        model = ImageRotations
        fields = ['img']
