import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    data_gen_args = dict(rescale=1./255, validation_split=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1

    image_generator = image_datagen.flow_from_directory(
        'wall_crack_dataset/train',
        classes=['crack', 'no_crack'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=32,
        subset='training',
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'wall_crack_dataset/masks/train',
        classes=['crack', 'no_crack'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=32,
        subset='training',
        seed=seed)

    val_image_generator = image_datagen.flow_from_directory(
        'wall_crack_dataset/val',
        classes=['crack', 'no_crack'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=32,
        subset='validation',
        seed=seed)

    val_mask_generator = mask_datagen.flow_from_directory(
        'wall_crack_dataset/masks/val',
        classes=['crack', 'no_crack'],
        class_mode=None,
        color_mode='grayscale',
        target_size=(256, 256),
        batch_size=32,
        subset='validation',
        seed=seed)

    def combine_generator(gen1, gen2):
        while True:
            yield (next(gen1), next(gen2))

    train_generator = combine_generator(image_generator, mask_generator)
    val_generator = combine_generator(val_image_generator, val_mask_generator)

    model = unet_model()
    checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_model.keras', save_best_only=True, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=image_generator.samples // image_generator.batch_size,
        validation_steps=val_image_generator.samples // val_image_generator.batch_size,
        epochs=10,  # Reduced from 50 to 10
        callbacks=[checkpoint, early_stopping]
    )

if __name__ == "__main__":
    main()
