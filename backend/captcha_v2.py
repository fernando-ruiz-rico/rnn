import tensorflow as tf
from tensorflow.keras import layers, models, datasets

import matplotlib
# Configuración esencial para entornos de servidor (headless). 
# 'Agg' permite generar imágenes en memoria sin requerir una interfaz gráfica (X11/GUI),
# evitando errores comunes al desplegar en contenedores o servidores remotos.
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

import numpy as np
import os
import time
from io import BytesIO

# Definición de rutas y constantes del sistema
NOMBRE_ARCHIVO_GENERADOR = "modelos/captcha_generador_v2_100e.keras"
NOMBRE_ARCHIVO_LECTOR = "modelos/captcha_lector_v2.keras"

BATCH_SIZE = 64
Z_SIZE = 100        # Dimensión del espacio latente (ruido) para el generador
EPOCHS_GAN = 50     # Ciclos de entrenamiento para la red adversaria
EPOCHS_LECTOR = 10  # Ciclos para el clasificador (necesita menos para converger en MNIST)

# Mapeo de índices a etiquetas legibles (Fashion MNIST)
NOMBRES_PRENDAS = [
    "Camiseta", "Pantalón", "Jersey", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota"
]

# Variables globales para mantener los modelos cargados en memoria
generador = None
lector = None

def verificar_hardware():
    """
    Comprobación de aceleración por hardware.
    Es recomendable verificar si CUDA está activo para el entrenamiento de la GAN,
    ya que es computacionalmente costoso.
    """
    if tf.config.list_physical_devices('GPU'):
        print("GPU disponible para TensorFlow.")
    else:
        print("GPU no disponible, usando CPU.")


def construir_lector():
    """
    Arquitectura CNN estándar para clasificación.
    Este modelo actúa como 'oráculo': etiquetará automáticamente las imágenes 
    que genere la GAN para que el sistema sepa cuál es la solución del CAPTCHA.
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Bloque convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Bloque convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Clasificación densa
        layers.Flatten(),
        layers.Dense(128, activation='relu'),

        layers.Dense(10, activation='softmax') # Salida probabilística para las 10 clases
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model           


def construir_generador():
    """
    Red Generadora (parte de la GAN).
    Toma un vector de ruido aleatorio y lo proyecta mediante convoluciones transpuestas
    (deconvoluciones) hasta formar una imagen de 28x28x1.
    """
    model = tf.keras.Sequential([
        # Proyección inicial y remodelado
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(Z_SIZE,)),
        layers.BatchNormalization(), # Normalización por lotes para estabilizar el gradiente
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),

        # Upsampling capa 1
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),    
        layers.LeakyReLU(),

        # Upsampling capa 2
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),    
        layers.LeakyReLU(),

        # Capa de salida: usamos tanh para obtener valores entre -1 y 1
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ]);

    return model

def construir_discriminador():
    """
    Red Discriminadora (parte de la GAN).
    Es un clasificador binario que intenta distinguir entre imágenes reales (del dataset)
    e imágenes falsas (creadas por el generador).
    """
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3), # Dropout para evitar sobreajuste rápido

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1) # Salida logit sin activación (se procesa en la función de pérdida)
    ])

    return model

def entrenar_y_guardar():
    """
    Flujo completo de entrenamiento.
    1. Entrena el 'Lector' (clasificador) con datos supervisados.
    2. Entrena la GAN (Generador vs Discriminador) con aprendizaje no supervisado.
    3. Serializa ambos modelos en disco.
    """
    if not os.path.exists("modelos"):
        os.makedirs("modelos")

    verificar_hardware()

    # Carga del dataset Fashion MNIST
    (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

    # Preprocesamiento 1: Para el Lector (Rango [0, 1])
    images_lector = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255.0

    # Preprocesamiento 2: Para la GAN (Rango [-1, 1])
    # Esto es estándar en GANs para alinearse con la activación 'tanh' del generador.
    images_gan = (train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') - 127.5) / 127.5

    # Pipeline de datos eficiente
    ds_gan = tf.data.Dataset.from_tensor_slices(images_gan).shuffle(60000).batch(BATCH_SIZE)

    # --- FASE 1: Entrenamiento del Lector ---
    print("Iniciando entrenamiento del Lector (Clasificador)...")
    lector = construir_lector()
    lector.fit(images_lector, train_labels, epochs=EPOCHS_LECTOR, batch_size=BATCH_SIZE)
    lector.save(NOMBRE_ARCHIVO_LECTOR)

    # --- FASE 2: Entrenamiento de la GAN ---
    print("Iniciando entrenamiento de la GAN...")
    generador = construir_generador()
    discriminador = construir_discriminador()

    # Función de pérdida: entropía cruzada binaria (Real vs Fake)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    # Decorador tf.function para compilar el paso de entrenamiento en un grafo estático (optimización)
    @tf.function
    def entrenar_paso(imagenes_reales):
        noise = tf.random.normal([BATCH_SIZE, Z_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generar imágenes falsas
            generated_images = generador(noise, training=True)

            # Evaluar imágenes
            real_output = discriminador(imagenes_reales, training=True)
            fake_output = discriminador(generated_images, training=True)

            # Cálculo de pérdidas
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output) # El generador quiere engañar al discriminador
            disc_loss = (cross_entropy(tf.ones_like(real_output), real_output) + \
                         cross_entropy(tf.zeros_like(fake_output), fake_output)) / 2 # El discriminador quiere acertar ambas
            
        # Cálculo y aplicación de gradientes
        grads_gen = gen_tape.gradient(gen_loss, generador.trainable_variables)
        grads_disc = disc_tape.gradient(disc_loss, discriminador.trainable_variables)

        gen_opt.apply_gradients(zip(grads_gen, generador.trainable_variables))
        disc_opt.apply_gradients(zip(grads_disc, discriminador.trainable_variables))

    # Bucle principal de entrenamiento GAN
    for epoch in range(EPOCHS_GAN):
        start = time.time()

        for imagenes_reales in ds_gan:
            entrenar_paso(imagenes_reales)

        print(f'Epoch {epoch + 1} / {EPOCHS_GAN}, tiempo: {time.time() - start:.2f} segundos')

    generador.save(NOMBRE_ARCHIVO_GENERADOR)
    return generador, lector

def cargar_o_entrenar():
    """
    Gestión de persistencia.
    Si los modelos ya existen en disco, se cargan para ahorrar tiempo.
    Si no, se inicia el proceso de entrenamiento.
    """
    if os.path.exists(NOMBRE_ARCHIVO_GENERADOR) and os.path.exists(NOMBRE_ARCHIVO_LECTOR):
        generador = tf.keras.models.load_model(NOMBRE_ARCHIVO_GENERADOR)
        lector = tf.keras.models.load_model(NOMBRE_ARCHIVO_LECTOR)
    else:
        generador, lector = entrenar_y_guardar()
    return generador, lector

def inicializar():
    global generador, lector
    generador, lector = cargar_o_entrenar()

def generar_captcha_v2():
    """
    Generación dinámica del CAPTCHA.
    1. Genera 4 imágenes sintéticas usando la GAN.
    2. Clasifica estas imágenes usando el Lector para obtener la solución correcta.
    3. Combina las imágenes en un único plot de Matplotlib.
    4. Devuelve el buffer de la imagen y las etiquetas.
    """
    if generador is None or lector is None:
        inicializar()

    cantidad_imagenes = 4
    # Generar ruido latente
    noise = tf.random.normal([cantidad_imagenes, Z_SIZE])
    # Inferencia del generador (training=False es importante para BatchNormalization)
    imagenes_falsas = generador(noise, training=False)
    
    # Re-escalar de [-1, 1] a [0, 1] para que el Lector las entienda
    imagenes_para_lector = (imagenes_falsas + 1) / 2.0

    # Obtener predicción de qué objetos son
    predicciones = lector.predict(imagenes_para_lector, verbose=0)
    indices_etiquetas = np.argmax(predicciones, axis=1)
    etiquetas = [NOMBRES_PRENDAS[i] for i in indices_etiquetas]

    # Renderizado de la imagen compuesta
    plt.figure(figsize=(10, 4))
    for i in range(cantidad_imagenes):
        plt.subplot(1, 4, i + 1)
        plt.imshow(imagenes_para_lector[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # Guardado en buffer de memoria (BytesIO) para evitar I/O de disco lento
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0.1)
    img.seek(0)
    plt.close() # Importante cerrar la figura para liberar memoria

    return img, etiquetas

# Carga inicial al importar el módulo
inicializar()