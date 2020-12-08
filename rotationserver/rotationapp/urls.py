from django.urls import path, include
from rest_framework import routers

from . import views

router = routers.DefaultRouter()
router.register('rotationapp', views.ImageRotationView)

urlpatterns = [
    path('', views.cxcontaxt, name='rotationform'),
    #path('api/', include(router.urls)),
]
