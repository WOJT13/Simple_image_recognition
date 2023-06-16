import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

ds = tf.keras.datasets.cifar10.load_data()
(training_images, training_labels), (testing_images, testing_labels) = ds
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Dear', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

option = int(input("1. Training 2. Testing\n"))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

if(option == 1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    model.save('image_classifier.model')

if(option == 2):
    model = tf.keras.models.load_model('image_classifier.model')

    img = cv.imread('plane.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.imshow(img, cmap=plt.cm.binary)

    prediction = model.predict(np.array([img]) /255)
    index = np.argmax(prediction)
    print(f'Prediction is {class_names[index]}')

    img2 = cv.imread('horse.jpg')
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

    plt.imshow(img2, cmap=plt.cm.binary)

    prediction2 = model.predict(np.array([img2])/255)
    index2 = np.argmax(prediction2)
    print(f'Prediction is {class_names[index2]}')