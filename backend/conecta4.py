import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuración de Reproducibilidad ---
# Establecemos semillas fijas para garantizar que los entrenamientos e inferencias
# sean deterministas y depurables.
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Constantes de configuración y rutas de persistencia del modelo
ARCHIVO_MODELO_C4 = 'modelos/conecta4_v2.keras'
ARCHIVO_DATOS_C4 = 'csv/conecta4_v2.csv'

FILAS = 6
COLUMNAS = 7
VACIO = 0
HUMANO = 1
IA = 2

# Variable global para mantener el modelo en memoria (Singleton pattern)
modelo = None

def es_movimiento_valido(tablero, columna):
    """Verifica si la columna no está llena revisando la fila superior."""
    return tablero[0][columna] == VACIO

def simular_jugada(tablero, columna, jugador):
    """
    Crea una copia del tablero y aplica un movimiento hipotético.
    Crucial para la evaluación heurística de "próximo movimiento ganador".
    """
    copia = tablero.copy()
    for f in range(FILAS - 1, -1, -1):
        if copia[f][columna] == VACIO:
            copia[f][columna] = jugador
            return copia
    return copia

def comprobar_victoria(tablero, jugador):
    """
    Algoritmo de comprobación de estado de victoria (4 en línea).
    Se usa tanto para entrenar como para la validación heurística en tiempo real.
    """
    # 1. Horizontal (-)
    for f in range(FILAS):
        for c in range(COLUMNAS - 3):
            if all(tablero[f][c+i] == jugador for i in range(4)): return True

    # 2. Vertical (|)
    for f in range(FILAS - 3):
        for c in range(COLUMNAS):
            if all(tablero[f+i][c] == jugador for i in range(4)): return True

    # 3. Diagonal Descendente (\)
    for f in range(FILAS - 3):
        for c in range(COLUMNAS - 3):
            if all(tablero[f+i][c+i] == jugador for i in range(4)): return True

    # 4. Diagonal Ascendente (/)
    for f in range(3, FILAS):
        for c in range(COLUMNAS - 3):
            if all(tablero[f-i][c+i] == jugador for i in range(4)): return True
            
    return False    

def cargar_datos():
    """
    Carga el dataset CSV. Se asume que las primeras 42 columnas son el tablero aplanado
    y la columna 43 es la etiqueta (movimiento óptimo).
    """
    if not os.path.exists(ARCHIVO_DATOS_C4):
        raise FileNotFoundError(f"El archivo {ARCHIVO_DATOS_C4} no existe.")
    
    # Shuffle de datos para romper correlaciones temporales
    df = pd.read_csv(ARCHIVO_DATOS_C4, header=None).sample(frac=1)

    X = df.iloc[:, 0:42].values.reshape((-1, 42))
    # One-hot encoding de la columna objetivo
    y = to_categorical(df.iloc[:, 42].values, num_classes=COLUMNAS)
    return X, y

def crear_modelo():
    """
    Definición de la arquitectura de la Red Neuronal.
    Uso de Embedding + LSTM Bidireccional: Tratamos el tablero como una secuencia
    de tokens (0, 1, 2) para capturar patrones espaciales complejos.
    """
    modelo_nuevo = Sequential([
        Input(shape=(42,)),
        # Embedding: Proyecta los valores discretos (0,1,2) a vectores densos de 16 dimensiones
        Embedding(input_dim=3, output_dim=16),
        # LSTM Bidireccional: Captura dependencias en ambas direcciones del tablero aplanado
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3), # Regularización para evitar overfitting
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        # Salida: Probabilidad de éxito para cada una de las 7 columnas
        Dense(COLUMNAS, activation='softmax')
    ])
    modelo_nuevo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo_nuevo

def obtener_modelo():
    """Gestiona la carga o el entrenamiento del modelo (Lazy Loading)."""
    global modelo
    
    # 1. Si ya está en memoria RAM, usarlo directamente.
    if modelo: return modelo

    # 2. Si existe el fichero guardado, cargarlo del disco.
    if os.path.exists(ARCHIVO_MODELO_C4):
        try:
            modelo = load_model(ARCHIVO_MODELO_C4)
            return modelo
        except: pass

    # 3. Si no existe modelo previo, entrenar desde cero.
    X, y = cargar_datos()
    modelo = crear_modelo()
    
    # EarlyStopping: Monitoriza la pérdida en validación. Si no mejora en 3 épocas,
    # detiene el entrenamiento y restaura los mejores pesos.
    parada = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    modelo.fit(X, y, epochs=25, batch_size=64, validation_split=0.2, callbacks=[parada], verbose=1)
    
    # Serialización del modelo para futuras ejecuciones
    os.makedirs(os.path.dirname(ARCHIVO_MODELO_C4), exist_ok=True)
    modelo.save(ARCHIVO_MODELO_C4)
    return modelo

def obtener_mejor_movimiento(tablero_lista):
    """
    Lógica principal de decisión de la IA.
    Utiliza un enfoque híbrido: Reglas deterministas + Inferencia de Red Neuronal.
    """
    tablero = np.array(tablero_lista).reshape(FILAS, COLUMNAS)
    validas = [c for c in range(COLUMNAS) if es_movimiento_valido(tablero, c)]

    # Si no hay movimientos, devolver error o código de empate (aquí simplificado a 0)
    if not validas: return 0

    # --- Heurística Defensiva/Ofensiva Prioritaria ---
    # Antes de consultar a la red, verificamos si podemos ganar o si debemos bloquear
    # una victoria inminente del rival.
    for jugador_objetivo in [IA, HUMANO]: 
        for col in validas:
            tablero_imaginario = simular_jugada(tablero, col, jugador_objetivo)
            
            if comprobar_victoria(tablero_imaginario, jugador_objetivo):
                return col
            
    # --- Inferencia Neuronal ---
    # Si no hay situaciones críticas inmediatas, usamos el modelo para predecir
    # el mejor movimiento estratégico a largo plazo.
    mi_modelo = obtener_modelo()
    # Predecir probabilidades para el estado actual
    prediccion = mi_modelo.predict(np.array([tablero_lista]), verbose=0)[0]

    # Ordenar columnas por probabilidad descendente
    ranking_columnas = np.argsort(prediccion)[::-1]

    # Seleccionar la mejor columna que sea legal
    for col in ranking_columnas:
        if col in validas:
            return int(col)
        
    # Fallback final (teóricamente inalcanzable si la lógica es correcta)
    return random.choice(validas)

# Inicialización temprana del modelo al importar el script
obtener_modelo()