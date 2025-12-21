import matplotlib
# Configuración del backend 'Agg' para renderizado sin interfaz gráfica (esencial para servidores/scripts)
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

# Configuración de semillas para garantizar la reproducibilidad determinista de los experimentos
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def obtener_datos(empresa, periodo='1y'):
    """
    Descarga los datos históricos de la empresa seleccionada.
    Gestiona la variabilidad en el formato de respuesta de yfinance (MultiIndex vs DataFrame plano).
    """
    df = yf.download(empresa, period=periodo, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No se encontraron datos para la empresa: {empresa}")

    # Manejo robusto para estructuras MultiIndex que devuelve versiones recientes de yfinance
    if isinstance(df.columns, pd.MultiIndex):
        try:
            serie = df.xs('Close', axis=1, level=0)
            if serie.empty:
                serie = df.xs(empresa, axis=1, level=1)['Close']
            return serie
        except:
            return df.iloc[:, 0]

    if 'Close' in df.columns:
        return df['Close']
    return df.iloc[:, 0]

def preparar_datos_para_rnn(datos, ventana=60):
    """
    Prepara la serie temporal para la red LSTM.
    1. Normaliza los datos (MinMax) para facilitar la convergencia del gradiente.
    2. Genera estructuras de ventana deslizante (Sliding Window) para aprendizaje supervisado.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Aseguramos formato columna para el escalador
    if isinstance(datos, pd.Series):
        datos_array = datos.values.reshape(-1, 1)
    else:
        datos_array = np.array(datos).reshape(-1, 1)

    scaler.fit(datos_array[:len(datos_array)])

    datos_escalados = scaler.transform(datos_array)

    # Transformación a problema supervisado:
    # X (Features): Secuencia de 'ventana' días.
    # y (Target): Valor del día siguiente.
    X, y = [], []
    for i in range(ventana, len(datos_escalados)):
        X.append(datos_escalados[i-ventana:i, 0])
        y.append(datos_escalados[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape necesario para LSTM: [Muestras, Pasos de tiempo, Características]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def crear_red_neuronal_rnn(input_shape):
    """
    Define la arquitectura de la red neuronal.
    Se utiliza una capa LSTM para capturar dependencias temporales a largo plazo.
    """
    modelo = Sequential([
        Input(shape=input_shape),
        LSTM(128), # Capa recurrente con 128 unidades
        Dense(1)   # Salida escalar (precio predicho)
    ]);

    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return modelo

def predecir_futuro(modelo, ultimos_datos_reales, scaler, dias_a_predecir=60):
    """
    Realiza predicciones recursivas fuera de la muestra (out-of-sample).
    La predicción de t+1 se usa como entrada para predecir t+2.
    Nota: Este método tiende a acumular error a medida que el horizonte de predicción aumenta.
    """
    predicciones = []
    secuencia_actual = ultimos_datos_reales.copy()

    for _ in range(dias_a_predecir):
        x_input = np.reshape(secuencia_actual, (1, len(secuencia_actual), 1))

        # Inferencia del modelo (verbose=0 para evitar saturar logs)
        prediccion_dia = modelo.predict(x_input, verbose=0)[0][0]
        predicciones.append(prediccion_dia)

        # Actualizamos la ventana deslizante eliminando el dato más antiguo y añadiendo la nueva predicción
        secuencia_actual = np.append(secuencia_actual[1:], [[prediccion_dia]], axis=0)

    # Inversión del escalado para obtener precios reales
    predicciones_reales = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1))

    return predicciones_reales.ravel()

def generar_grafico_predicciones(empresa='AAPL', periodo='1y', epochs=50):
    try:
        datos_cierre = obtener_datos(empresa, periodo)

        print(datos_cierre)

        ventana = 60
        # Preprocesamiento
        X, y, scaler = preparar_datos_para_rnn(datos_cierre, ventana)

        # Instanciación y entrenamiento
        modelo = crear_red_neuronal_rnn((X.shape[1], 1))

        # Shuffle=False es importante en series temporales si no se usan ventanas independientes,
        # aunque aquí con ventanas explícitas es menos crítico, mantenemos el orden.
        modelo.fit(X, y, epochs=epochs, batch_size=32, verbose=0, shuffle=False)

        # Inferencia futura
        ultimos_60_dias = scaler.transform(datos_cierre.values[-ventana:].reshape(-1, 1))
        predicciones = predecir_futuro(modelo, ultimos_60_dias, scaler)

        # Generación de la gráfica
        plt.figure(figsize=(10, 6.5))

        datos_recientes = datos_cierre.iloc[-120:]
        plt.plot(datos_recientes.index, datos_recientes.values, label='Datos Reales', color='b')

        ultima_fecha = datos_recientes.index[-1]
        fechas_futuras = pd.date_range(start=ultima_fecha, periods=len(predicciones)+1)[1:]

        plt.plot(fechas_futuras, predicciones, label='Predicciones Futuras', color='r', linestyle='--')

        plt.title(f'Predicción de Precios para {empresa}', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel('Precio', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Guardado en buffer de memoria para evitar I/O de disco innecesario
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return img
    
    except Exception as e:
        print(f"Error crítico al generar la gráfica de predicciones: {e}")
        return None