import numpy as np
import pandas as pd
import os
import random
import re
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # Sequential: poner capas una tras otra. load_model: cargar uno ya entrenado.
from tensorflow.keras.callbacks import EarlyStopping       # Para detener el entrenamiento si la IA deja de mejorar (ahorra tiempo).
from tensorflow.keras.layers import Embedding, LSTM, Dense # Tipos de neuronas: Embedding (traductor), LSTM (memoria), Dense (decisión final).
from tensorflow.keras.preprocessing.text import Tokenizer  # Convierte palabras en números (la IA solo entiende números).
from tensorflow.keras.preprocessing.sequence import pad_sequences # Rellena frases cortas para que todas midan lo mismo.
from tensorflow.keras.utils import to_categorical          # Convierte las respuestas en un formato de "probabilidades" (0s y 1s).

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

ARCHIVO_MODELO = 'modelos/texto_predictivo_modelo.keras'
ARCHIVO_TOKENIZADOR = 'modelos/texto_predictivo_tokenizador.pkl'
ARCHIVO_DATOS = 'csv/texto_predictivo.csv'

tokenizer = None
modelo = None
max_sequence_length = 20  # Longitud máxima de las secuencias de entrada

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto


def cargar_datos(ruta_csv):
    if not os.path.exists(ruta_csv):
        print(f"El archivo {ruta_csv} no existe.")
        return ['hola cómo estás', 'buenos días', 'buenas noches', 'estoy bien gracias', 'buen día para ti también', 'que descanses']
    
    df = pd.read_csv(ruta_csv)
    data = df['frase'].dropna().drop_duplicates().tolist()
    return [limpiar_texto(frase) for frase in data]


def preparar_datos(frases):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(frases)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in frases:
        token_list = tokenizer.texts_to_sequences([line])[0] # Convierte ["hola", "como"] en [4, 12]
        # Creamos n-gramas: "hola como" -> predice "estas"
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_len = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

    xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
    ys = to_categorical(labels, num_classes=total_words)

    return xs, ys, max_len, total_words, tokenizer


def crear_modelo(total_words, embedding_dim, input_length):
    modelo = Sequential([
        Embedding(total_words, embedding_dim, input_length=input_length),
        LSTM(128),
        Dense(total_words, activation='softmax')
    ]);
    modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelo


def predecir_texto(semilla, num_palabras=10):
    texto_actual = limpiar_texto(semilla)

    siguiente_palabra = predecir_proxima_palabra(texto_actual)

    for _ in range(num_palabras):
        siguiente_palabra = predecir_proxima_palabra(texto_actual)

        if siguiente_palabra:
            ultima_palabra = texto_actual.split()[-1]
            if siguiente_palabra != ultima_palabra:
                texto_actual += ' ' + siguiente_palabra
            else:
                break
        else:
            break

    return texto_actual.capitalize()


def predecir_proxima_palabra(texto):
    token_list = tokenizer.texts_to_sequences([texto])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    
    prediccion = modelo.predict(token_list, verbose=0)[0]
    indice_ganador = np.argmax(prediccion)

    if prediccion[indice_ganador] < 0.1:
        return None

    return tokenizer.index_word[indice_ganador]


def inicializar():
    global modelo, tokenizer, max_sequence_length

    if os.path.exists(ARCHIVO_MODELO) and os.path.exists(ARCHIVO_TOKENIZADOR):
        modelo = load_model(ARCHIVO_MODELO)
        with open(ARCHIVO_TOKENIZADOR, 'rb') as handle:
            datos = pickle.load(handle)
            tokenizer = datos['tokenizer']
            max_sequence_length = datos['max_len']

    else:
        frases = cargar_datos(ARCHIVO_DATOS)
        tokenizer, total_words, max_sequence_length, xs, ys = preparar_datos(frases)

        modelo = crear_modelo(total_words, 128, max_sequence_length-1)

        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        modelo.fit(xs, ys, epochs=50, verbose=1, shuffle=True, callbacks=[early_stop])

        modelo.save(ARCHIVO_MODELO)

        paquete = {'tokenizer': tokenizer, 'max_len': max_sequence_length}
        with open(ARCHIVO_TOKENIZADOR, 'wb') as handle:
            pickle.dump(paquete, handle, protocol=pickle.HIGHEST_PROTOCOL)

inicializar()