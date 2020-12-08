import PIL.Image
import keras
import numpy as np
from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render
from rest_framework import viewsets, response, status

from .forms import RotationForm
from .models import ImageRotations
from .serializers import ImageRotationsSerializers


class ImageRotationView(viewsets.ModelViewSet):
    queryset = ImageRotations.objects.all()
    serializer_class = ImageRotationsSerializers


def rotation(img):
    img_path = f'{settings.MEDIA_ROOT}\\images\\{img}'
    new_img_path = f'{settings.MEDIA_ROOT}\\rotated\\{img}'
    try:
        nn = keras.models.load_model('../nn/models/augmentation_model')
        image = PIL.Image.open(img_path)
        values = preprocess_image(image)
        y_pred = np.argmax(nn.predict(values))
        if y_pred == 1:
            text_pred = '90 Degrees'
            image_as_array = np.array(image)
            image_as_array = np.rot90(image_as_array, 3)
            temp_image = PIL.Image.fromarray(image_as_array)
            temp_image.save(new_img_path)
        elif y_pred == 2:
            text_pred = '180 degrees'
            image_as_array = np.array(image)
            image_as_array = np.rot90(image_as_array, 2)
            temp_image = PIL.Image.fromarray(image_as_array)
            temp_image.save(new_img_path)
        elif y_pred == 3:
            text_pred = '-90 Degrees'
            image_as_array = np.array(image)
            image_as_array = np.rot90(image_as_array, 1)
            temp_image = PIL.Image.fromarray(image_as_array)
            temp_image.save(new_img_path)
        else:
            text_pred = 'Correct'
            image.save(new_img_path)

        return text_pred
    except ValueError as e:
        return response.Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def preprocess_image(image):
    ratio = min(28 / image.width, 28 / image.height)
    aspect_ratio = image.width / image.height
    resized_image = image.resize((int(image.width * ratio),
                                  int(image.height * ratio)))
    print(f"resized: h - {resized_image.height} / w - {resized_image.width}")
    values = np.array(resized_image.convert('L')) / 255
    if aspect_ratio > 1:
        cols = 28 - resized_image.height
        first_col_copy = values[0, :]
        last_col_copy = values[-1, :]
        for i in range(cols):
            if i % 2 == 0:
                values = np.insert(values, 0, first_col_copy, axis=0)
            else:
                values = np.insert(values, -1, last_col_copy, axis=0)
    elif aspect_ratio < 1:
        cols = 28 - resized_image.width
        first_col_copy = values[:, 0]
        last_col_copy = values[:, -1]
        for i in range(cols):
            if i % 2 == 0:
                values = np.insert(values, 0, first_col_copy, axis=1)
            else:
                values = np.insert(values, -1, last_col_copy, axis=1)
    else:
        print("Image was given in correct shape :)")

    values = values.reshape(1, 28, 28, 1)
    return values


def cxcontaxt(request):
    if request.method == "POST":
        form = RotationForm(request.POST, request.FILES)
        if form.is_valid():
            img = f"{request.FILES['img']}"
            print(f"{img}")
            form.save()
            path = f'{settings.MEDIA_ROOT}\\rotated\\{img}'
            rote = rotation(img)
            messages.success(request, f"Rotation: {rote}")

            try:
                with open(path, 'rb') as f:
                    if "png" in img:
                        res = HttpResponse(f.read(), content_type='image/png')
                    else:
                        res = HttpResponse(f.read(), content_type='image/jpeg')
                    res['Content-Disposition'] = f'attachment; filename={img}'
            except IOError:
                red = PIL.Image.new('RGB', (1, 1), (255, 0, 0))
                res = HttpResponse(content_type="image/png")
                red.save(res, 'PNG')

            return res
        else:
            print(f"Error: {form.errors}")
    form = RotationForm()

    return render(request, 'rotationapp/cxform.html', {'form': form})
