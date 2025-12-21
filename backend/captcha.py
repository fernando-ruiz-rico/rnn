import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# [CONFIGURACIÓN MATPLOTLIB]
# Es fundamental usar el backend 'Agg' en entornos de servidor (headless).
# Esto evita que Python intente abrir ventanas gráficas (GUI) al generar plots,
# lo cual fallaría en un contenedor o servidor web.
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

import numpy as np
import os
import time
from io import BytesIO

# Constantes de configuración y rutas de persistencia del modelo.
NOMBRE_ARCHIVO_GENERADOR = "modelos/captcha_generador_100e.keras"
NOMBRE_ACCHIVO_LECTOR = "modelos/captcha_lector_100e.keras"

# Hiperparámetros de entrenamiento
BATCH_SIZE = 64
Z_SIZE = 100     # Dimensión del espacio latente (ruido de entrada para la GAN)
EPOCHS_GAN = 100
EPOCHS_LECTOR = 10

# Variables globales para mantener los modelos cargados en memoria RAM
generador = None
lector = None

def verificar_hardware():
    """
    Verificación de disponibilidad de aceleración por hardware (CUDA/GPU).
    Crítico para el rendimiento durante la fase de entrenamiento (backpropagation).
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: print("GPU detectada. Entrenamiento acelerado habilitado.")
    else: print("NO se detectó GPU. El entrenamiento se ejecutará en CPU (lento).")


def construir_lector():
    """
    Define la arquitectura de la Red Neuronal Convolucional (CNN) para clasificación.
    Esta red actuará como el 'solucionador' del captcha generado.
    Arquitectura estándar: Conv2D -> MaxPool -> Conv2D -> MaxPool -> Dense.
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax') # Salida: Probabilidad para 10 dígitos (0-9)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def construir_generador():
    """
    Define el Generador de la GAN (DCGAN).
    Transforma ruido aleatorio (vector Z) en una imagen de 28x28x1.
    Utiliza Conv2DTranspose para realizar el 'upsampling' de las características.
    Salida: Activación 'tanh' para normalizar píxeles entre [-1, 1].
    """
    model = tf.keras.Sequential([
        # Entrada: Espacio latente -> Proyección Densa
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(Z_SIZE,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        # Reshape a mapa de características 7x7
        layers.Reshape((7, 7, 256)),
        
        # Upsampling 1: 7x7 -> 7x7 (misma dimensión, más profundidad de filtros)
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        # Upsampling 2: 7x7 -> 14x14
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        # Upsampling 3: 14x14 -> 28x28 (Tamaño final MNIST)
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model


def construir_discriminador():
    """
    Define el Discriminador de la GAN.
    Es un clasificador binario que intenta distinguir entre imágenes reales (MNIST)
    e imágenes falsas generadas por el modelo anterior.
    """
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3), # Regularización para evitar sobreajuste rápido
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1) # Salida: Logit (Real vs Fake)
    ])
    return model


def entrenar_y_guardar():
    """
    Rutina principal de orquestación del entrenamiento.
    1. Preprocesa el dataset MNIST.
    2. Entrena el 'Lector' (CNN clásica).
    3. Entrena la GAN (Generador vs Discriminador) mediante un bucle de entrenamiento personalizado.
    4. Serializa los modelos en disco (.keras).
    """
    if not os.path.exists('modelos'):
        os.makedirs('modelos')

    verificar_hardware()

    # Carga del dataset MNIST
    (train_images, train_labels), (_, _) = datasets.mnist.load_data()

    # [NORMALIZACIÓN 1 - Lector] El lector espera [0, 1]
    images_lector = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0

    # [NORMALIZACIÓN 2 - GAN] El generador usa tanh, por lo que las imágenes reales deben estar en [-1, 1]
    images_gan = (train_images.reshape((60000, 28, 28, 1)).astype('float32') - 127.5) / 127.5

    # Pipeline de datos eficiente con tf.data
    ds_gan = tf.data.Dataset.from_tensor_slices(images_gan).shuffle(60000).batch(BATCH_SIZE)

    print("--- Iniciando entrenamiento del modelo Lector (CNN) ---")
    lector = construir_lector()
    lector.fit(images_lector, train_labels, epochs=EPOCHS_LECTOR, batch_size=BATCH_SIZE)
    lector.save(NOMBRE_ACCHIVO_LECTOR)

    print("--- Iniciando entrenamiento de la GAN (Generador + Discriminador) ---")
    generador = construir_generador()
    discriminador = construir_discriminador()

    # Función de pérdida: Entropía cruzada binaria (Log Loss)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    # Paso de entrenamiento optimizado con @tf.function (Grafo estático)
    @tf.function
    def train_step(real_images):
        noise = tf.random.normal([BATCH_SIZE, Z_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generador(noise, training=True)

            real_output = discriminador(real_images, training=True)
            fake_output = discriminador(generated_images, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

            # Loss del discriminador: penaliza fallos en reales y fallos en fakes
            disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                        cross_entropy(tf.zeros_like(fake_output), fake_output)

        grads_gen = gen_tape.gradient(gen_loss, generador.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss, discriminador.trainable_variables)

        gen_opt.apply_gradients(zip(grads_gen, generador.trainable_variables))
        disc_opt.apply_gradients(zip(grads_disc, discriminador.trainable_variables))

    # Bucle de entrenamiento GAN
    for epoch in range(EPOCHS_GAN):
        start = time.time()

        for image_batch in ds_gan:
            train_step(image_batch)

        print(f'Epoch {epoch + 1}, tiempo: {time.time() - start:.2f} segundos')

    generador.save(NOMBRE_ARCHIVO_GENERADOR)
    return generador, lector

def cargar_o_entrenar():
    """
    Estrategia de persistencia: Si los modelos ya existen en disco, se cargan para
    ahorrar tiempo. Si no, se dispara el entrenamiento completo.
    """
    if os.path.exists(NOMBRE_ARCHIVO_GENERADOR) and os.path.exists(NOMBRE_ACCHIVO_LECTOR):
        print("Modelos pre-entrenados detectados. Cargando desde disco...")
        generador = tf.keras.models.load_model(NOMBRE_ARCHIVO_GENERADOR)
        lector = tf.keras.models.load_model(NOMBRE_ACCHIVO_LECTOR)
    else:
        print("Modelos no encontrados. Iniciando secuencia de entrenamiento...")
        generador, lector = entrenar_y_guardar()
    return generador, lector

def inicializar():
    global generador, lector
    generador, lector = cargar_o_entrenar()

def generar_captcha():
    """
    Genera un captcha compuesto por 4 dígitos sintéticos.
    Retorna la imagen combinada (buffer) y el código numérico resuelto por el Lector.
    """
    if generador is None or lector is None:
        inicializar()

    # Generamos 4 vectores de ruido aleatorio
    ruido = tf.random.normal([4, Z_SIZE])

    # El generador crea las imágenes (valores en rango [-1, 1])
    imagenes_falsas = generador(ruido, training=False)

    # [POST-PROCESADO] Reescalado a [0, 1] para que el Lector pueda interpretarlas correctamente
    imagenes_para_lector = (imagenes_falsas + 1) / 2.0

    # Inferencia: El Lector predice qué números ha creado el Generador
    predicciones = lector.predict(imagenes_para_lector, verbose=0)
    etiquetas = np.argmax(predicciones, axis=1)

    # Construcción de la solución (string)
    codigo_secreto = ''.join(str(digito) for digito in etiquetas)
    
    # Composición de la imagen final usando Matplotlib
    plt.figure(figsize=(10, 4))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        # Seleccionamos el canal 0 ya que es escala de grises
        plt.imshow(imagenes_para_lector[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # Guardado en buffer de memoria (IO) para no escribir en disco
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    return img, codigo_secreto

# Arranque inicial al importar el módulo
inicializar()