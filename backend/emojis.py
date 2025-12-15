import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stemmer = SnowballStemmer('spanish')

def preprocesar_texto(text):
    texto = re.sub(r'\W', ' ', text)
    texto = texto.lower()
    tokens = word_tokenize(texto)
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('spanish')]
    return ' '.join(tokens)

def enternar_modelo(csv_path):
    df = pd.read_csv(csv_path)

    df = df.drop_duplicates(subset='emojis')
    df['palabras'] = df['palabras'].apply(preprocesar_texto)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X = vectorizer.fit_transform(df['palabras'])

    modelo = MultinomialNB().fit(X, df['emojis'])

    return {'modelo': modelo, 'vectorizer': vectorizer}

def traducir_a_emojis(modelo_emojis, texto, num_emojis=5):
    texto = preprocesar_texto(texto)

    modelos = modelo_emojis['modelo']
    vectorizer = modelo_emojis['vectorizer']

    X_test = vectorizer.transform([texto])

    probabilidades = modelos.predict_proba(X_test)[0]
    indices_ordenados = probabilidades.argsort()[::-1][:num_emojis]

    resultado = []
    for idx in indices_ordenados:
        emoji = modelos.classes_[idx]
        similitud = round(probabilidades[idx] * 100, 2)

        if similitud > 0.10:
            resultado.append({'emojis': emoji, 'similitud': similitud})

    return resultado

modelo_emojis = enternar_modelo('csv/emojis.csv')
