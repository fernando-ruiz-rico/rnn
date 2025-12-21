import tensorflow as tf
from tensorflow.keras import layers, models

import matplotlib

# Configuración del backend 'Agg' para Matplotlib.
# Esto es crítico en entornos de servidor (headless) donde no hay pantalla física disponible.
# Evita errores de runtime relacionados con Tcl/Tk al intentar renderizar ventanas emergentes.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import os
import time
from io import BytesIO

# Constantes de configuración y rutas de persistencia del modelo
NOMBRE_ARCHIVO_GENERADOR = "modelos/moda_generador_150e.keras"

BATCH_SIZE = 64
# Dimensión del espacio latente (vector de ruido). 100 es un estándar habitual en GANs simples.
Z_SIZE = 100 
EPOCHS_GAN = 50

generador_global = None

def construir_generador():
    """
    Define la arquitectura del Generador basada en una DCGAN (Deep Convolutional GAN).
    Objetivo: Transformar un vector de ruido (Z_SIZE) en una imagen de 28x28x1.
    """
    model = tf.keras.Sequential([
        # Entrada: Vector de ruido. Proyección densa y remodelado para iniciar las convoluciones.
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(Z_SIZE,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),

        # Upsampling bloque 1: De 7x7 a 7x7 (padding 'same', stride 1) pero reduciendo profundidad.
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),    
        layers.LeakyReLU(),

        # Upsampling bloque 2: De 7x7 a 14x14.
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),    
        layers.LeakyReLU(),

        # Capa de salida: De 14x14 a 28x28 (tamaño original Fashion MNIST).
        # Usamos 'tanh' para normalizar la salida en el rango [-1, 1].
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ]);

    return model

def construir_discriminador():
    """
    Define la arquitectura del Discriminador (Clasificador Binario).
    Objetivo: Distinguir entre imágenes reales del dataset y falsas generadas.
    """
    model = models.Sequential([
        # Downsampling mediante strides=2 en lugar de MaxPooling (práctica recomendada en DCGANs).
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3), # Dropout para prevenir el sobreajuste y estabilizar el entrenamiento.

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1) # Salida logit sin activación (la sigmoide se gestiona en la función de pérdida).
    ])

    return model

def entrenar_y_guardar():
    """
    Ejecuta el bucle de entrenamiento completo si no existe un modelo previo.
    Gestiona la carga de datos, normalización y el ciclo de GradientTape.
    """
    if not os.path.exists("modelos"):
        os.makedirs("modelos")

    # Carga del dataset Fashion MNIST
    (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

    # Preprocesamiento: Normalización a [-1, 1] para coincidir con la activación 'tanh' del generador.
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    # Pipeline de datos optimizado con shuffle y batching
    ds_train = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(BATCH_SIZE)

    generador = construir_generador()
    discriminador = construir_discriminador()

    # Función de pérdida: BinaryCrossentropy con from_logits=True para mayor estabilidad numérica.
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    # Decorador tf.function para compilar el paso de entrenamiento en un grafo estático (mayor rendimiento).
    @tf.function
    def entrenar_paso(imagenes_reales):
        noise = tf.random.normal([BATCH_SIZE, Z_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 1. El generador crea imágenes falsas
            generated_images = generador(noise, training=True)

            # 2. El discriminador evalúa imágenes reales y falsas
            real_output = discriminador(imagenes_reales, training=True)
            fake_output = discriminador(generated_images, training=True)

            # 3. Cálculo de pérdidas
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output) # El generador quiere engañar al discriminador
            
            # Pérdida del discriminador: promedio de error en reales y error en falsas
            disc_loss = (cross_entropy(tf.ones_like(real_output), real_output) + \
                         cross_entropy(tf.zeros_like(fake_output), fake_output)) / 2
            
        # 4. Cálculo y aplicación de gradientes
        grads_gen = gen_tape.gradient(gen_loss, generador.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss, discriminador.trainable_variables)

        gen_opt.apply_gradients(zip(grads_gen, generador.trainable_variables))
        disc_opt.apply_gradients(zip(grads_disc, discriminador.trainable_variables))

    # Bucle principal de épocas
    for epoch in range(EPOCHS_GAN):
        start = time.time()

        for imagenes_reales in ds_train:
            entrenar_paso(imagenes_reales)

        print(f'Epoch {epoch + 1} / {EPOCHS_GAN}, tiempo: {time.time() - start:.2f} segundos')

    # Persistencia del modelo generador para inferencia futura
    generador.save(NOMBRE_ARCHIVO_GENERADOR)
    return generador

def cargar_o_entrenar():
    """
    Estrategia de caché en disco: Intenta cargar el modelo pre-entrenado.
    Si falla o no existe, inicia el reentrenamiento.
    """
    if os.path.exists(NOMBRE_ARCHIVO_GENERADOR):
        try:
            generador = tf.keras.models.load_model(NOMBRE_ARCHIVO_GENERADOR)
            return generador
        except Exception as e:
            print(f"Error al cargar el modelo: {e}. Entrenando un nuevo modelo...")
    return entrenar_y_guardar()

def inicializar():
    """
    Singleton para mantener el generador en memoria y evitar recargas en cada petición.
    """
    global generador_global
    if generador_global is None:
        generador_global = cargar_o_entrenar()
    return generador_global

def generar_muestra_ropa():
    """
    Genera un grid de 3x3 imágenes sintéticas y lo devuelve como un buffer de bytes (PNG).
    Ideal para servir directamente a través de una API Flask/Django.
    """
    if generador_global is None:
        inicializar()

    num_ejemplos = 9
    ruido = tf.random.normal([num_ejemplos, Z_SIZE])

    # Inferencia: training=False es crucial para que BatchNormalization use estadísticas acumuladas y no del batch actual.
    predicciones = generador_global(ruido, training=False)

    fig = plt.figure(figsize=(5, 5))

    for i in range(predicciones.shape[0]):
        plt.subplot(3, 3, i + 1)
        # Desnormalizar de [-1, 1] a [0, 1] para visualización correcta
        imagen_visualizable = (predicciones[i, :, :, 0] + 1) / 2.0

        plt.imshow(imagen_visualizable, cmap='gray')
        plt.axis('off')

    # Guardado en buffer de memoria (BytesIO) en lugar de disco para optimizar I/O
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0.1)
    img_buf.seek(0)
    plt.close(fig) # Liberar memoria de Matplotlib explícitamente
    return img_buf

inicializar()