import tensorflow as tf
from tensorflow.keras import layers, models

import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

import os
import time
from io import BytesIO

NOMBRE_ARCHIVO_GENERADOR = "modelos/moda_generador_150e.keras"

BATCH_SIZE = 64
Z_SIZE = 100
EPOCHS_GAN = 50

generador_global = None

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
    ]);

    return model

def construir_discriminador():
    model = models.Sequential([
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
    if not os.path.exists("modelos"):
        os.makedirs("modelos")

    (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    ds_train = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(BATCH_SIZE)

    generador = construir_generador()
    discriminador = construir_discriminador()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def entrenar_paso(imagenes_reales):
        noise = tf.random.normal([BATCH_SIZE, Z_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generador(noise, training=True)

            real_output = discriminador(imagenes_reales, training=True)
            fake_output = discriminador(generated_images, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

            disc_loss = (cross_entropy(tf.ones_like(real_output), real_output) + \
                         cross_entropy(tf.zeros_like(fake_output), fake_output)) / 2
            
        grads_gen = gen_tape.gradient(gen_loss, generador.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss, discriminador.trainable_variables)

        gen_opt.apply_gradients(zip(grads_gen, generador.trainable_variables))
        disc_opt.apply_gradients(zip(grads_disc, discriminador.trainable_variables))

    for epoch in range(EPOCHS_GAN):
        start = time.time()

        for imagenes_reales in ds_train:
            entrenar_paso(imagenes_reales)

        print(f'Epoch {epoch + 1} / {EPOCHS_GAN}, tiempo: {time.time() - start:.2f} segundos')

    generador.save(NOMBRE_ARCHIVO_GENERADOR)
    return generador

def cargar_o_entrenar():
    if os.path.exists(NOMBRE_ARCHIVO_GENERADOR):
        try:
            generador = tf.keras.models.load_model(NOMBRE_ARCHIVO_GENERADOR)
            return generador
        except Exception as e:
            print(f"Error al cargar el modelo: {e}. Entrenando un nuevo modelo...")
    return entrenar_y_guardar()

def inicializar():
    global generador_global
    if generador_global is None:
        generador_global = cargar_o_entrenar()
    return generador_global

def generar_muestra_ropa():
    if generador_global is None:
        inicializar()

    num_ejemplos = 9
    ruido = tf.random.normal([num_ejemplos, Z_SIZE])

    predicciones = generador_global(ruido, training=False)

    fig = plt.figure(figsize=(5, 5))

    for i in range(predicciones.shape[0]):
        plt.subplot(3, 3, i + 1)
        imagen_visualizable = (predicciones[i, :, :, 0] + 1) / 2.0

        plt.imshow(imagen_visualizable, cmap='gray')
        plt.axis('off')

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0.1)
    img_buf.seek(0)
    plt.close(fig)
    return img_buf

inicializar()