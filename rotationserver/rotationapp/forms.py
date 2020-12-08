from django import forms

from .models import ImageRotations


class RotationForm(forms.ModelForm):
    clothing_image = forms.ImageField(required=False)

    class Meta:
        model = ImageRotations
        fields = ('clothing_image',)
