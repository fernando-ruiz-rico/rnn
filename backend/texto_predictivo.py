import numpy as np
import pandas as pd
import os
import random
import re
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # Gestión del modelo secuencial y carga de persistencia.
from tensorflow.keras.callbacks import EarlyStopping       # Callback para evitar overfitting y reducir tiempo de cómputo.
from tensorflow.keras.layers import Embedding, LSTM, Dense # Capas: Vectorización semántica (Embedding), Memoria recurrente (LSTM) y Clasificador (Dense).
from tensorflow.keras.preprocessing.text import Tokenizer  # Utilidad para vectorización de texto (índices numéricos).
from tensorflow.keras.preprocessing.sequence import pad_sequences # Normalización de la longitud de los tensores de entrada.
from tensorflow.keras.utils import to_categorical          # Codificación One-Hot para las etiquetas de salida.

# --- Configuración de Reproducibilidad ---
# Fijamos las semillas aleatorias para garantizar que el entrenamiento sea determinista.
# Esto asegura que obtengamos los mismos resultados en diferentes ejecuciones con los mismos datos.
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Constantes de configuración y rutas de persistencia del modelo
ARCHIVO_MODELO = 'modelos/texto_predictivo_modelo.keras'
ARCHIVO_TOKENIZADOR = 'modelos/texto_predictivo_tokenizador.pkl'
ARCHIVO_DATOS = 'csv/texto_predictivo.csv'

# Variables globales para mantener el estado del modelo y tokenizador en memoria
tokenizer = None
modelo = None
max_sequence_length = 20  # Límite superior para la ventana de contexto

def limpiar_texto(texto):
    """
    Normalización del texto: conversión a minúsculas y eliminación de caracteres especiales
    para reducir la dimensionalidad del vocabulario.
    """
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto


def cargar_datos(ruta_csv):
    """
    Carga el dataset desde un CSV o devuelve un corpus por defecto (fallback) si el archivo no existe.
    Realiza una limpieza preliminar y elimina duplicados.
    """
    if not os.path.exists(ruta_csv):
        print(f"Aviso: El archivo {ruta_csv} no se encuentra. Usando corpus de fallback.")
        return ['hola cómo estás', 'buenos días', 'buenas noches', 'estoy bien gracias', 'buen día para ti también', 'que descanses']
    
    df = pd.read_csv(ruta_csv)
    # Eliminamos nulos y duplicados para evitar sesgos en el entrenamiento
    data = df['frase'].dropna().drop_duplicates().tolist()
    return [limpiar_texto(frase) for frase in data]


def preparar_datos(frases):
    """
    Transforma el corpus de texto en secuencias numéricas (tensores) aptas para la red neuronal.
    Utiliza una técnica de ventana deslizante para generar N-gramas.
    Ej: "hola como estas" -> [hola, como], [hola, como, estas]
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(frases)
    total_words = len(tokenizer.word_index) + 1 # +1 por el token de padding (índice 0)

    input_sequences = []
    for line in frases:
        token_list = tokenizer.texts_to_sequences([line])[0] 
        
        # Generación de secuencias de entrenamiento (N-gramas)
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Padding: Rellenamos con ceros a la izquierda ('pre') para que todas las secuencias tengan la misma longitud
    max_len = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

    # Separación en Features (X) y Labels (y)
    # X: Todos los tokens menos el último.
    # y: El último token (la palabra a predecir).
    xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
    
    # Convertimos las etiquetas a formato categórico (One-Hot Encoding)
    ys = to_categorical(labels, num_classes=total_words)

    return xs, ys, max_len, total_words, tokenizer


def crear_modelo(total_words, embedding_dim, input_length):
    """
    Definición de la arquitectura de la Red Neuronal Recurrente (RNN).
    1. Embedding: Transforma índices enteros en vectores densos.
    2. LSTM: Capa de memoria a largo plazo para capturar contexto secuencial.
    3. Dense (Softmax): Capa de salida que asigna probabilidades a cada palabra del vocabulario.
    """
    modelo = Sequential([
        Embedding(total_words, embedding_dim, input_length=input_length),
        LSTM(128), # 128 unidades de memoria
        Dense(total_words, activation='softmax')
    ]);
    # Optimizador Adam y función de pérdida Categorical Crossentropy (estándar para clasificación multiclase)
    modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelo


def predecir_texto(semilla, num_palabras=10):
    """
    Genera texto de forma iterativa basándose en una semilla inicial.
    Utiliza un enfoque voraz (greedy) seleccionando la palabra con mayor probabilidad en cada paso.
    """
    texto_actual = limpiar_texto(semilla)

    siguiente_palabra = predecir_proxima_palabra(texto_actual)

    for _ in range(num_palabras):
        siguiente_palabra = predecir_proxima_palabra(texto_actual)

        if siguiente_palabra:
            # Control básico para evitar bucles infinitos de la misma palabra
            ultima_palabra = texto_actual.split()[-1]
            if siguiente_palabra != ultima_palabra:
                texto_actual += ' ' + siguiente_palabra
            else:
                break
        else:
            break

    return texto_actual.capitalize()


def predecir_proxima_palabra(texto):
    """
    Realiza la inferencia de una única palabra dado un contexto.
    Incluye un umbral de confianza para descartar predicciones débiles.
    """
    token_list = tokenizer.texts_to_sequences([texto])[0]
    # Aplicamos el mismo padding que en el entrenamiento
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    
    # Inferencia (verbose=0 para silenciar logs de Keras)
    prediccion = modelo.predict(token_list, verbose=0)[0]
    indice_ganador = np.argmax(prediccion)

    # Umbral de certidumbre: si la confianza es menor al 10%, no predecimos nada
    if prediccion[indice_ganador] < 0.1:
        return None

    return tokenizer.index_word[indice_ganador]


def inicializar():
    """
    Rutina de arranque (Bootstrapping).
    Verifica si existe un modelo pre-entrenado para cargarlo. 
    Si no, inicia el pipeline de entrenamiento completo: carga, procesamiento, entrenamiento y persistencia.
    """
    global modelo, tokenizer, max_sequence_length

    # Comprobamos persistencia para evitar re-entrenar en cada reinicio
    if os.path.exists(ARCHIVO_MODELO) and os.path.exists(ARCHIVO_TOKENIZADOR):
        print("Cargando modelo y tokenizador pre-entrenados...")
        modelo = load_model(ARCHIVO_MODELO)
        with open(ARCHIVO_TOKENIZADOR, 'rb') as handle:
            datos = pickle.load(handle)
            tokenizer = datos['tokenizer']
            max_sequence_length = datos['max_len']

    else:
        print("Iniciando entrenamiento desde cero...")
        frases = cargar_datos(ARCHIVO_DATOS)
        print(f"Corpus cargado: {len(frases)} frases.")

        xs, ys, max_sequence_length, total_words, tokenizer = preparar_datos(frases)

        # Ajustamos la longitud de entrada restando 1 porque la última palabra es la etiqueta (y)
        modelo = crear_modelo(total_words, 128, max_sequence_length-1)

        # EarlyStopping: Detiene el entrenamiento si la función de pérdida no mejora en 5 épocas
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        modelo.fit(xs, ys, epochs=50, verbose=1, shuffle=True, callbacks=[early_stop])

        # Serialización (guardado) del modelo y artefactos
        modelo.save(ARCHIVO_MODELO)

        paquete = {'tokenizer': tokenizer, 'max_len': max_sequence_length}
        with open(ARCHIVO_TOKENIZADOR, 'wb') as handle:
            pickle.dump(paquete, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Ejecución inicial
inicializar()