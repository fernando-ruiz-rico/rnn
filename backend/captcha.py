import tensorflow as tf
from tensorflow.keras import layers, models, datasets

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts
import matplotlib.pyplot as plt

import numpy as np
import os
import time
from io import BytesIO

NOMBRE_ARCHIVO_GENERADOR = "modelos/captcha_generador.keras"
NOMBRE_ACCHIVO_LECTOR = "modelos/captcha_lector.keras"

BATCH_SIZE = 64
Z_SIZE = 100
EPOCHS_GAN = 100
EPOCHS_LECTOR = 10

generador = None
lector = None

def verificar_hardware():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: print("GPU detectada.")
    else: print("NO se detectó GPU.")


def construir_lector():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def construir_generador():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(Z_SIZE,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model


def construir_discriminador():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


def entrenar_y_guardar():
    if not os.path.exists('modelos'):
        os.makedirs('modelos')

    verificar_hardware()

    (train_images, train_labels), (_, _) = datasets.mnist.load_data()

    images_lector = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0

    images_gan = (train_images.reshape((60000, 28, 28, 1)).astype('float32') - 127.5) / 127.5

    ds_gan = tf.data.Dataset.from_tensor_slices(images_gan).shuffle(60000).batch(BATCH_SIZE)

    print("Entrenando lector...")
    lector = construir_lector()
    lector.fit(images_lector, train_labels, epochs=EPOCHS_LECTOR, batch_size=BATCH_SIZE)
    lector.save(NOMBRE_ACCHIVO_LECTOR)

    print("Entrenando GAN...")
    generador = construir_generador()
    discriminador = construir_discriminador()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(real_images):
        noise = tf.random.normal([BATCH_SIZE, Z_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generador(noise, training=True)

            real_output = discriminador(real_images, training=True)
            fake_output = discriminador(generated_images, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

            disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                        cross_entropy(tf.zeros_like(fake_output), fake_output)

        grads_gen = gen_tape.gradient(gen_loss, generador.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss, discriminador.trainable_variables)

        gen_opt.apply_gradients(zip(grads_gen, generador.trainable_variables))
        disc_opt.apply_gradients(zip(grads_disc, discriminador.trainable_variables))

    for epoch in range(EPOCHS_GAN):
        start = time.time()

        for image_batch in ds_gan:
            train_step(image_batch)

        print(f'Epoch {epoch + 1}, tiempo: {time.time() - start:.2f} segundos')

    generador.save(NOMBRE_ARCHIVO_GENERADOR)
    return generador, lector

def cargar_o_entrenar():
    if os.path.exists(NOMBRE_ARCHIVO_GENERADOR) and os.path.exists(NOMBRE_ACCHIVO_LECTOR):
        print("Cargando modelos existentes...")
        generador = tf.keras.models.load_model(NOMBRE_ARCHIVO_GENERADOR)
        lector = tf.keras.models.load_model(NOMBRE_ACCHIVO_LECTOR)
    else:
        print("Modelos no encontrados. Iniciando entrenamiento...")
        generador, lector = entrenar_y_guardar()
    return generador, lector

def inicializar():
    global generador, lector
    generador, lector = cargar_o_entrenar()

def generar_captcha():
    if generador is None or lector is None:
        inicializar()

    ruido = tf.random.normal([4, Z_SIZE])

    imagenes_falsas = generador(ruido, training=False)

    imagenes_para_lector = (imagenes_falsas + 1) / 2.0

    predicciones = lector.predict(imagenes_para_lector, verbose=0)

    etiquetas = np.argmax(predicciones, axis=1)

    codigo_secreto = ''.join(str(digito) for digito in etiquetas)
    
    plt.figure(figsize=(10, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(imagenes_para_lector[i, :, :, 0], cmap='gray')
        plt.axis('off')

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    return img, codigo_secreto

inicializar()