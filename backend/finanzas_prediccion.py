import matplotlib
# Configuración del backend 'Agg' para renderizado sin interfaz gráfica.
# Fundamental para evitar errores de 'TclError' o 'RuntimeError' al ejecutar Matplotlib
# en servidores web o contenedores Docker donde no hay un monitor conectado.
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from io import BytesIO
import os
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# --- Configuración de Determinismo ---
# Las redes neuronales, especialmente en GPU, tienden a ser no deterministas debido a la ejecución paralela.
# Fijamos las semillas para intentar que los resultados sean reproducibles durante el desarrollo y depuración.
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1' # Fuerza operaciones deterministas en cuDNN (si se usa GPU NVIDIA)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def obtener_datos(empresa, periodo='1y'):
    """
    Descarga los datos históricos de la empresa seleccionada.
    Implementa lógica defensiva contra cambios en la API de yfinance.
    """
    # 'auto_adjust=True' es crucial en finanzas para descontar dividendos y splits del precio.
    df = yf.download(empresa, period=periodo, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No se encontraron datos bursátiles para el ticker: {empresa}")

    # Manejo robusto para estructuras MultiIndex.
    # Versiones recientes de yfinance devuelven índices jerárquicos (Ticker -> OHLCV).
    # Este bloque asegura que siempre obtenemos una Serie unidimensional limpia.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            serie = df.xs('Close', axis=1, level=0)
            if serie.empty:
                serie = df.xs(empresa, axis=1, level=1)['Close']
            return serie
        except:
            # Fallback de emergencia: retornamos la primera columna si la estructura es desconocida
            return df.iloc[:, 0]

    if 'Close' in df.columns:
        return df['Close']
    return df.iloc[:, 0]

def preparar_datos_para_rnn(datos, ventana=60):
    """
    Transforma la serie temporal en un dataset supervisado para entrenamiento de RNNs.
    
    Args:
        datos: Serie temporal de precios.
        ventana: 'Look-back period'. Cuántos pasos pasados usa la IA para predecir el siguiente.
        
    Returns:
        X: Tensor de entrada [Muestras, Pasos de tiempo, Características].
        y: Vector objetivo (valor esperado).
        scaler: Objeto normalizador (necesario para des-normalizar la predicción después).
    """
    # Las LSTMs son muy sensibles a la escala de los datos. 
    # Normalizamos entre 0 y 1 para facilitar la convergencia del descenso de gradiente.
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Aseguramos formato columna (matriz 2D) para el escalador de Scikit-Learn
    if isinstance(datos, pd.Series):
        datos_array = datos.values.reshape(-1, 1)
    else:
        datos_array = np.array(datos).reshape(-1, 1)

    scaler.fit(datos_array[:len(datos_array)]) # Ajustamos el escalador a los datos disponibles

    datos_escalados = scaler.transform(datos_array)

    # Creación de ventanas deslizantes (Sliding Window).
    # Si ventana=60:
    # X[0] = días 0 a 59 -> y[0] = día 60
    # X[1] = días 1 a 60 -> y[1] = día 61
    X, y = [], []
    for i in range(ventana, len(datos_escalados)):
        X.append(datos_escalados[i-ventana:i, 0])
        y.append(datos_escalados[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape crítico para Keras/TensorFlow.
    # Una capa LSTM espera un tensor 3D con la forma: [Batch Size, Time Steps, Features].
    # Aquí Features es 1 porque solo usamos una variable (el precio de cierre).
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def crear_red_neuronal_rnn(input_shape):
    """
    Arquitectura de la Red Neuronal Recurrente (RNN).
    Se opta por LSTM (Long Short-Term Memory) por su capacidad para evitar el problema
    del desvanecimiento del gradiente en secuencias temporales largas.
    """
    modelo = Sequential([
        Input(shape=input_shape),
        # 128 neuronas en la capa oculta permiten capturar patrones complejos.
        # return_sequences=False (por defecto) porque solo queremos la salida tras el último paso de tiempo.
        LSTM(128), 
        # Capa densa de salida con 1 neurona lineal para regresión (predecir un valor continuo).
        Dense(1)   
    ]);

    # Optimizador Adam: Estándar de la industria por su manejo adaptativo del learning rate.
    # Loss MSE (Mean Squared Error): Penaliza más los errores grandes, adecuado para predicción de precios.
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return modelo

def predecir_futuro(modelo, ultimos_datos_reales, scaler, dias_a_predecir=60):
    """
    Realiza predicciones 'fuera de muestra' de forma recursiva (Autoregressive Inference).
    
    IMPORTANTE: La predicción del día t+1 se convierte en entrada para t+2.
    Esto introduce un sesgo acumulativo: cualquier error pequeño en la primera predicción
    se magnifica exponencialmente a medida que nos alejamos en el tiempo.
    """
    predicciones = []
    secuencia_actual = ultimos_datos_reales.copy()

    for _ in range(dias_a_predecir):
        # Redimensionamos la ventana actual para que coincida con la entrada esperada por el modelo (1, 60, 1)
        x_input = np.reshape(secuencia_actual, (1, len(secuencia_actual), 1))

        # Inferencia. verbose=0 suprime la barra de progreso para mantener limpios los logs del servidor.
        prediccion_dia = modelo.predict(x_input, verbose=0)[0][0]
        predicciones.append(prediccion_dia)

        # Actualización de la ventana deslizante:
        # Eliminamos el dato más antiguo (índice 0) y añadimos la nueva predicción al final.
        secuencia_actual = np.append(secuencia_actual[1:], [[prediccion_dia]], axis=0)

    # Invertimos la normalización para devolver precios en la escala monetaria original ($)
    predicciones_reales = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1))

    return predicciones_reales.ravel() # Aplanamos a array 1D

def generar_grafico_predicciones(empresa='AAPL', periodo='1y', epochs=50):
    """
    Orquesta el flujo completo ETL -> Entrenamiento -> Inferencia -> Visualización.
    Maneja excepciones para asegurar que el servidor no se detenga ante errores de modelado.
    """
    try:
        # 1. Obtención de datos
        datos_cierre = obtener_datos(empresa, periodo)

        # Logs de depuración para verificar la integridad de los datos en consola
        print(f"Datos obtenidos para {empresa}: {len(datos_cierre)} registros.")

        ventana = 60
        # 2. Preprocesamiento y creación de tensores
        X, y, scaler = preparar_datos_para_rnn(datos_cierre, ventana)

        # 3. Construcción del modelo
        modelo = crear_red_neuronal_rnn((X.shape[1], 1))

        # 4. Entrenamiento
        # shuffle=False mantiene el orden de los lotes. Aunque en este diseño de ventanas independientes
        # no es estrictamente obligatorio, es buena práctica en series temporales para conservar cierta cronología.
        modelo.fit(X, y, epochs=epochs, batch_size=32, verbose=0, shuffle=False)

        # 5. Inferencia (Predicción futura)
        # Tomamos los últimos 60 días reales para arrancar la cadena de predicciones
        ultimos_60_dias = scaler.transform(datos_cierre.values[-ventana:].reshape(-1, 1))
        predicciones = predecir_futuro(modelo, ultimos_60_dias, scaler)

        # 6. Generación del gráfico
        plt.figure(figsize=(10, 6.5))

        # Mostramos solo los últimos 120 días reales para tener contexto visual cercano
        datos_recientes = datos_cierre.iloc[-120:]
        plt.plot(datos_recientes.index, datos_recientes.values, label='Datos Reales', color='b')

        # Generamos el eje de fechas futuras
        ultima_fecha = datos_recientes.index[-1]
        fechas_futuras = pd.date_range(start=ultima_fecha, periods=len(predicciones)+1)[1:]

        plt.plot(fechas_futuras, predicciones, label='Predicciones Futuras (IA)', color='r', linestyle='--')

        plt.title(f'Predicción de Precios (LSTM) para {empresa}', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel('Precio', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Guardado en buffer de memoria (optimización de I/O)
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Cierre explícito de la figura para liberar RAM en el servidor
        plt.close()

        return img
    
    except Exception as e:
        # Captura de errores críticos (ej: problemas de convergencia, datos insuficientes)
        print(f"Error crítico al generar la gráfica de predicciones: {e}")
        return None