import pandas as pd
import matplotlib
# Se establece el backend 'Agg' para evitar que Matplotlib intente abrir una ventana GUI
# en el servidor, lo cual generaría un error en tiempo de ejecución en entornos headless.
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts

import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO

def obtener_datos(empresa, periodo='1y'):
    # Descarga de datos bursátiles. Se activa auto_adjust para obtener precios ajustados por dividendos/splits.
    df = yf.download(empresa, period=periodo, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No se encontraron datos para la empresa: {empresa}")

    # Gestión de robustez para diferentes versiones de yfinance o estructuras de datos devueltas.
    # A veces devuelve un MultiIndex (Price, Ticker) y otras un índice simple.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Intento de acceso directo al nivel 0 si la estructura es jerárquica
            serie = df.xs('Close', axis=1, level=0)
            if serie.empty:
                # Fallback: acceso por nivel de empresa si el anterior falla
                serie = df.xs(empresa, axis=1, level=1)['Close']
            return serie
        except:
            # Si falla la navegación por índices, devolvemos la primera columna disponible por seguridad
            return df.iloc[:, 0]

    # Estructura estándar de una sola dimensión
    if 'Close' in df.columns:
        return df['Close']
    return df.iloc[:, 0]

def generar_grafico_empresas(empresa, periodo):
    datos = obtener_datos(empresa, periodo)
    
    fechas = []
    precios_cierre = []

    # Iteración explícita para formateo de fechas. 
    # Nota: Aunque se podría vectorizar con pandas, este bucle asegura el control total 
    # sobre el formato de salida string para el eje X.
    for fecha in datos.index:
        fecha_obj = fecha.to_pydatetime()

        fecha_formateada = fecha_obj.strftime('%d %b %Y')

        fechas.append(fecha_formateada)
        precios_cierre.append(datos.loc[fecha])

    # Configuración de dimensiones del lienzo (canvas)
    plt.figure(figsize=(10, 6.5))

    plt.plot(fechas, precios_cierre, label='Precio de Cierre', color='b')

    plt.title(f'Evolución de {empresa}', fontsize=16)

    plt.xlabel('Fecha', fontsize=14)

    # Obtención dinámica de la divisa para el etiquetado correcto del eje Y
    moneda = yf.Ticker(empresa).info['currency']
    plt.ylabel(f'Precio ({moneda})', fontsize=14)

    # Lógica para evitar saturación de etiquetas en el eje X
    num_etiquetas = 10
    step = max(1, len(fechas) // num_etiquetas)
    plt.xticks(fechas[::step], rotation=45, ha='right')

    plt.grid(True)

    plt.tight_layout()

    # Uso de buffer en memoria (BytesIO) para guardar la imagen.
    # Esto evita operaciones de I/O en disco, mejorando el rendimiento y la limpieza del servidor.
    img = BytesIO()
    plt.savefig(img, format='png')

    # Rebobinado del puntero al inicio del stream para su posterior lectura
    img.seek(0)

    # Es fundamental cerrar la figura para liberar memoria en el servidor tras cada petición
    plt.close()

    return img