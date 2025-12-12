import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for scripts

import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO

def obtener_datos(empresa, periodo='1y'):
    df = yf.download(empresa, period=periodo, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No se encontraron datos para la empresa: {empresa}")

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

def generar_grafico_empresas(empresa, periodo):
    datos = obtener_datos(empresa, periodo)
    
    fechas = []
    precios_cierre = []

    for fecha in datos.index:
        fecha_obj = fecha.to_pydatetime()

        fecha_formateada = fecha_obj.strftime('%d %b %Y')

        fechas.append(fecha_formateada)
        precios_cierre.append(datos.loc[fecha, 'Close'])

    plt.figure(figsize=(10, 6.5))

    plt.plot(fechas, precios_cierre, label='Precio de Cierre', color='b')

    plt.title(f'Evolución de {empresa}', fontsize=16)

    plt.xlabel('Fecha', fontsize=14)

    moneda = yf.Ticker(empresa).info['currency']
    plt.ylabel(f'Precio ({moneda})', fontsize=14)

    num_etiquetas = 10
    step = max(1, len(fechas) // num_etiquetas)
    plt.xticks(fechas[::step], rotation=45, ha='right')

    plt.grid(True)

    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')

    img.seek(0)

    plt.close()

    return img
