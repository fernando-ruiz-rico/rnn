import pandas as pd
import matplotlib
# Configuración del backend 'Agg' (Anti-Grain Geometry).
# Es fundamental en entornos de servidor (headless) donde no existe una pantalla física.
# Esto evita que Matplotlib intente instanciar ventanas GUI, lo que provocaría un crash de la aplicación.
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO

def obtener_datos(empresa, periodo='1y'):
    """
    Recupera los datos históricos de cotización para un ticker específico.
    Gestiona la variabilidad en la estructura de datos devuelta por la API de Yahoo Finance.
    """
    # Descarga de datos bursátiles. 
    # 'auto_adjust=True' es crítico para normalizar el precio de cierre teniendo en cuenta 
    # eventos corporativos como splits y dividendos, ofreciendo una visión real del retorno.
    df = yf.download(empresa, period=periodo, progress=False, auto_adjust=True)
    
    # Validación temprana para evitar errores en cadena si el ticker no existe o no hay conexión
    if df.empty:
        raise ValueError(f"No se encontraron datos para la empresa: {empresa}")

    # --- Gestión de Robustez para Estructuras de Datos ---
    # La librería yfinance actualiza frecuentemente su formato de retorno.
    # A veces devuelve un DataFrame plano y otras un MultiIndex (especialmente si se descargan múltiples tickers
    # o en versiones recientes). Este bloque asegura que siempre extraigamos la serie temporal correcta.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Estrategia 1: Acceso jerárquico estándar (Nivel 0: Tipo de dato, Nivel 1: Ticker)
            serie = df.xs('Close', axis=1, level=0)
            if serie.empty:
                # Estrategia 2: Inversión de niveles (Nivel 0: Ticker, Nivel 1: Tipo de dato)
                serie = df.xs(empresa, axis=1, level=1)['Close']
            return serie
        except:
            # Fallback de seguridad: Si la navegación por índices falla, asumimos que la primera 
            # columna contiene la información relevante para no detener la ejecución.
            return df.iloc[:, 0]

    # Caso estándar: DataFrame de una sola dimensión con columna explícita 'Close'
    if 'Close' in df.columns:
        return df['Close']
    
    # Último recurso: devolver la primera columna disponible
    return df.iloc[:, 0]

def generar_grafico_empresas(empresa, periodo):
    """
    Genera una visualización de la evolución del precio de las acciones.
    Renderiza el gráfico en un buffer de memoria para su envío directo al cliente web.
    """
    datos = obtener_datos(empresa, periodo)
    
    fechas = []
    precios_cierre = []

    # Iteración explícita para formateo de fechas. 
    # Aunque Pandas permite vectorización, este bucle nos da control granular sobre 
    # la conversión de Timestamp a string para asegurar la compatibilidad con el eje X de Matplotlib
    # y el formato visual deseado ('Día Mes Año').
    for fecha in datos.index:
        fecha_obj = fecha.to_pydatetime()
        fecha_formateada = fecha_obj.strftime('%d %b %Y')
        fechas.append(fecha_formateada)
        precios_cierre.append(datos.loc[fecha])

    # Inicialización del lienzo (canvas) con dimensiones específicas para web
    plt.figure(figsize=(10, 6.5))

    # Trazado de la serie temporal
    plt.plot(fechas, precios_cierre, label='Precio de Cierre', color='b')

    # Decoración del gráfico
    plt.title(f'Evolución de {empresa}', fontsize=16)
    plt.xlabel('Fecha', fontsize=14)

    # Obtención dinámica de metadatos (divisa) para enriquecer la información del eje Y.
    # Nota: Esto implica una llamada de red adicional a la API.
    moneda = yf.Ticker(empresa).info['currency']
    plt.ylabel(f'Precio ({moneda})', fontsize=14)

    # --- Optimización de la Legibilidad del Eje X ---
    # Evitamos la saturación de etiquetas (overcrowding) mostrando solo una subselección.
    # Calculamos un 'step' dinámico para mostrar un máximo aproximado de 10 etiquetas.
    num_etiquetas = 10
    step = max(1, len(fechas) // num_etiquetas)
    plt.xticks(fechas[::step], rotation=45, ha='right')

    plt.grid(True) # Añadimos rejilla para facilitar la lectura de valores
    plt.tight_layout() # Ajuste automático de márgenes para evitar cortes en las etiquetas

    # --- Gestión de Memoria y Salida ---
    # Uso de buffer en memoria (BytesIO) en lugar de guardar en disco.
    # Esto es más eficiente (evita I/O de disco) y limpio (no genera archivos temporales basura),
    # siendo la práctica estándar para servir imágenes dinámicas en aplicaciones web.
    img = BytesIO()
    plt.savefig(img, format='png')

    # Rebobinado del puntero al inicio del stream para que pueda ser leído posteriormente
    img.seek(0)

    # CIERRE DE LA FIGURA: Paso crítico en servidores web.
    # Matplotlib mantiene las figuras en memoria globalmente. Si no se cierran explícitamente con plt.close(),
    # el servidor sufrirá una fuga de memoria (memory leak) y acabará colapsando tras muchas peticiones.
    plt.close()

    return img