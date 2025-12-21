import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Descarga de recursos necesarios de NLTK para tokenización y listado de palabras vacías (stopwords).
# Es necesario ejecutarlo al menos una vez en el entorno de despliegue.
nltk.download('punkt')
nltk.download('stopwords')

# Inicializamos el Stemmer en español.
# El objetivo es reducir las palabras a su raíz (ej: "corriendo" -> "corr") para reducir la dimensionalidad del vector.
stemmer = SnowballStemmer('spanish')

def preprocesar_texto(text):
    """
    Realiza la normalización y limpieza del texto de entrada.
    Elimina caracteres no alfanuméricos, convierte a minúsculas y aplica stemming.
    """
    # Eliminación de ruido: caracteres especiales fuera del rango alfanumérico.
    texto = re.sub(r'\W', ' ', text)
    texto = texto.lower()
    
    # Tokenización y filtrado de stopwords (palabras sin carga semántica como 'el', 'de', 'la').
    tokens = word_tokenize(texto)
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('spanish')]
    
    return ' '.join(tokens)

def enternar_modelo(csv_path):
    """
    Entrena el modelo de clasificación supervisada Multinomial Naive Bayes.
    Utiliza TF-IDF para la extracción de características, considerando n-gramas.
    """
    df = pd.read_csv(csv_path)

    # Limpieza de duplicados basada en la columna 'emojis' para evitar sesgos en el entrenamiento si hay clases desbalanceadas.
    # Nota: Dependiendo del dataset, podría interesar mantener duplicados si reflejan mayor frecuencia de uso.
    df = df.drop_duplicates(subset='emojis')
    
    # Aplicamos la normalización a todo el corpus de entrenamiento.
    df['palabras'] = df['palabras'].apply(preprocesar_texto)

    # Configuración del Vectorizador TF-IDF (Term Frequency - Inverse Document Frequency).
    # ngram_range=(1, 3) permite capturar contexto local (ej: "no me gusta" vs "me gusta").
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    
    # Transformación del corpus a una matriz dispersa de características.
    X = vectorizer.fit_transform(df['palabras'])

    # Entrenamiento del modelo. MultinomialNB funciona eficientemente con conteos discretos (como frecuencias de palabras).
    modelo = MultinomialNB().fit(X, df['emojis'])

    return {'modelo': modelo, 'vectorizer': vectorizer}

def traducir_a_emojis(modelo_emojis, texto, num_emojis=5):
    """
    Realiza la inferencia sobre un texto nuevo para sugerir emojis relevantes.
    Devuelve los N emojis más probables que superen un umbral de confianza.
    """
    texto = preprocesar_texto(texto)

    modelos = modelo_emojis['modelo']
    vectorizer = modelo_emojis['vectorizer']

    # Transformamos el texto de entrada al mismo espacio vectorial que el conjunto de entrenamiento.
    X_test = vectorizer.transform([texto])

    # Obtenemos las probabilidades de cada clase en lugar de solo la predicción final.
    probabilidades = modelos.predict_proba(X_test)[0]
    
    # Ordenamos los índices de mayor a menor probabilidad y tomamos los top N.
    indices_ordenados = probabilidades.argsort()[::-1][:num_emojis]

    resultado = []
    for idx in indices_ordenados:
        emoji = modelos.classes_[idx]
        similitud = round(probabilidades[idx] * 100, 2)

        # Filtro de corte (Threshold): Solo devolvemos resultados con una confianza superior al 10%
        # para evitar ruido o sugerencias irrelevantes.
        if similitud > 0.10:
            resultado.append({'emojis': emoji, 'similitud': similitud})

    return resultado

# Carga e inicialización del modelo al arrancar el script.
modelo_emojis = enternar_modelo('csv/emojis.csv')