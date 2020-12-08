import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def rotate_image(img, degree):
    if degree == 0:
        return img
    elif degree == 90:
        return np.rot90(img)
    elif degree == 180:
        return np.rot90(img, 2)
    elif degree == 270:
        return np.rot90(img, 3)
    else:
        print("degree not supported")


class_names = ['0', '90', '180', '-90']

(train_images, _), (
    test_images, _) = tf.keras.datasets.fashion_mnist.load_data()

train_images = 1 - train_images / 255
test_images = 1 - test_images / 255

train_labels = []
test_labels = []

for i in range(len(train_images)):
    train_images[i] = rotate_image(train_images[i], 90 * (i % 4))
    train_labels.append(i % 4)

for i in range(len(test_images)):
    test_images[i] = rotate_image(test_images[i], 90 * (i % 4))
    test_labels.append(i % 4)

train_images_reshaped = train_images.reshape(
    (train_images.shape[0], 28, 28, 1))
test_images_reshaped = test_images.reshape((test_images.shape[0], 28, 28, 1))

train_labels_onehot = keras.utils.to_categorical(train_labels, 4)
test_labels_onehot = keras.utils.to_categorical(test_labels, 4)

x_train, x_validation, y_train, y_validation = train_test_split(
    train_images_reshaped, train_labels_onehot, test_size=0.1, random_state=42)


def make_model():
    return keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4, activation='softmax')
    ])


model = make_model()

augmentation = ImageDataGenerator(shear_range=0.1,
                                  zoom_range=0.1,
                                  rotation_range=15)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(augmentation.flow(x_train, y_train, batch_size=100),
                    validation_data=(x_validation, y_validation),
                    epochs=3)

model.save('./models/augmentation_model')
# model = keras.models.load_model('./models/basic_model')

print(model.evaluate(test_images_reshaped, test_labels_onehot))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.get_cmap('Greys'))
    plt.xlabel(f"Prediction:"
               f"{class_names[np.argmax(model.predict(test_images[i].reshape(1,28,28,1)))]} "
               f"Label:{class_names[test_labels[i]]}")
    temp_image = Image.fromarray(test_images[i] * 255)
    temp_image = temp_image.convert('RGB')
    temp_image.save(f'../rsc/clothing{i}.png')

plt.show()
