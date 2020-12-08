import PIL.Image
import keras
import numpy as np
import os
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework import viewsets, response, status

from .forms import RotationForm
from .models import ImageRotations
from .serializers import ImageRotationsSerializers


class ImageRotationView(viewsets.ModelViewSet):
    queryset = ImageRotations.objects.all()
    serializer_class = ImageRotationsSerializers


def cxcontaxt(request):
    if request.method == "POST":
        form = RotationForm(request.POST, request.FILES)
        if form.is_valid():
            image_name = request.FILES.get('clothing_image', False)
            if image_name:
                form.save()
                path_rotated = f'{settings.MEDIA_ROOT}\\rotated\\{image_name}'
                rotate(image_name)
                os.remove(f'{settings.MEDIA_ROOT}\\images\\{image_name}')
                try:
                    with open(path_rotated, 'rb') as image_file:
                        if "png" in image_name:
                            res = HttpResponse(image_file.read(),
                                               content_type='image/png')
                        else:
                            res = HttpResponse(image_file.read(),
                                               content_type='image/jpeg')
                        res['Content-Disposition'] = \
                            f'attachment; filename={image_name}'
                        return res
                except IOError:
                    form = RotationForm()
                    return render(request, 'rotationapp/cxform.html',
                                  {'form': form})
        else:
            print(f"Error: {form.errors}")
    form = RotationForm()

    return render(request, 'rotationapp/cxform.html', {'form': form})


def rotate(img):
    img_path = f'{settings.MEDIA_ROOT}\\images\\{img}'
    new_img_path = f'{settings.MEDIA_ROOT}\\rotated\\{img}'
    try:
        nn = keras.models.load_model('../nn/models/custom_augmentation_model')
        image = PIL.Image.open(img_path)
        values = preprocess_image(image)
        y_pred = np.argmax(nn.predict(values))
        if y_pred == 1:
            # 90 Degrees
            image_as_array = np.array(image)
            image_as_array = np.rot90(image_as_array, 3)
            temp_image = PIL.Image.fromarray(image_as_array)
            temp_image.save(new_img_path)
        elif y_pred == 2:
            # 180 degrees
            image_as_array = np.array(image)
            image_as_array = np.rot90(image_as_array, 2)
            temp_image = PIL.Image.fromarray(image_as_array)
            temp_image.save(new_img_path)
        elif y_pred == 3:
            # -90 Degrees
            image_as_array = np.array(image)
            image_as_array = np.rot90(image_as_array, 1)
            temp_image = PIL.Image.fromarray(image_as_array)
            temp_image.save(new_img_path)
        else:
            # Correct
            image.save(new_img_path)

    except ValueError as e:
        print(e)


def preprocess_image(image):
    ratio = min(28 / image.width, 28 / image.height)
    resized_image = image.resize((int(image.width * ratio),
                                  int(image.height * ratio)))

    values = np.array(resized_image.convert('L')) / 255

    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        values = vertical_padding(resized_image, values)
    elif aspect_ratio < 1:
        values = horizontal_padding(resized_image, values)

    values = values.reshape(1, 28, 28, 1)

    return values


def horizontal_padding(resized_image, values):
    cols = 28 - resized_image.width
    first_col_copy = values[:, 0]
    last_col_copy = values[:, -1]
    for i in range(cols):
        if i % 2 == 0:
            values = np.insert(values, 0, first_col_copy, axis=1)
        else:
            values = np.insert(values, -1, last_col_copy, axis=1)
    return values


def vertical_padding(resized_image, values):
    cols = 28 - resized_image.height
    first_col_copy = values[0, :]
    last_col_copy = values[-1, :]
    for i in range(cols):
        if i % 2 == 0:
            values = np.insert(values, 0, first_col_copy, axis=0)
        else:
            values = np.insert(values, -1, last_col_copy, axis=0)
    return values

