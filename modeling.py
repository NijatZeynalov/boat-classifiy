import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "split_boats//training//"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "split_boats//testing//"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

model = tf.keras.models.Sequential()

# 1st Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters=256, input_shape=(150,150,3), kernel_size=(11,11),activation='relu' ,strides=(4,4), padding='valid'))
# Pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(tf.keras.layers.BatchNormalization())

# 2nd Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(tf.keras.layers.Activation('relu'))
# Pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# 3rd Convolutional Layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(tf.keras.layers.Activation('relu'))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# Passing it to a dense layer
model.add(tf.keras.layers.Flatten())
# 1st Dense Layer
model.add(tf.keras.layers.Dense(16, input_shape=(224*224*3,)))
model.add(tf.keras.layers.Activation('relu'))
# Add Dropout to prevent overfitting
model.add(tf.keras.layers.Dropout(0.4))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# 2nd Dense Layer
model.add(tf.keras.layers.Dense(48))
model.add(tf.keras.layers.Activation('relu'))
# Add Dropout
model.add(tf.keras.layers.Dropout(0.4))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# 3rd Dense Layer
model.add(tf.keras.layers.Dense(24))
model.add(tf.keras.layers.Activation('relu'))
# Add Dropout
model.add( tf.keras.layers.Dropout(0.4))
# Batch Normalisation
model.add(tf.keras.layers.BatchNormalization())

# Output Layer
model.add(tf.keras.layers.Dense(9))
model.add(tf.keras.layers.Activation('softmax'))

model.summary()

# (4) Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_generator, epochs=30, steps_per_epoch=100, validation_steps = 50, validation_data = validation_generator, verbose = 2)

model.save("boat_model.h5")

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)

plt.show()