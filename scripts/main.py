import json
import nltk
import os
from datetime import datetime  # Importar para el nombre único de archivos
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import matplotlib.pyplot as plt

# Descargar recursos de NLTK
nltk.download('wordnet', download_dir='/content/nltk_data/')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.data.load("tokenizers/punkt/english.pickle")
nltk.download('stopwords')

# Función para preprocesar texto
def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    tokens = word_tokenize(text)  # Tokenización
    tokens = [token for token in tokens if token.isalnum()]  # Eliminar no alfanuméricos
    stop_words = set(stopwords.words('english'))  # Palabras irrelevantes
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lematización
    return " ".join(tokens)

# Cargar reseñas desde el archivo JSONL
def load_reviews(file_path):
    reviews = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                review = json.loads(line.strip())
                reviews.append(review["text"])
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en la ruta {file_path}.")
    except KeyError:
        print("Error: No se encontró la clave 'text' en algunas reseñas.")
    return reviews

# Analizar sentimientos usando un modelo preentrenado de Transformers
def analyze_sentiments(reviews):
    print("Iniciando análisis de sentimientos...")
    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    sentiments = []
    for review in reviews:
        result = classifier(review)[0]
        sentiments.append(result["label"])  # Etiquetas como "1 star", "5 stars"
    return sentiments

# Categorizar sentimientos
def categorize_sentiments(sentiments):
    categories = {"Insatisfecho": 0, "Neutral": 0, "Satisfecho": 0}
    for sentiment in sentiments:
        if "1 star" in sentiment or "2 stars" in sentiment:
            categories["Insatisfecho"] += 1
        elif "3 stars" in sentiment:
            categories["Neutral"] += 1
        elif "4 stars" in sentiment or "5 stars" in sentiment:
            categories["Satisfecho"] += 1
    return categories

# Graficar resultados y guardar imagen
def plot_results(categories):
    print("Generando gráfica...")
    
    # Crear la gráfica
    plt.bar(categories.keys(), categories.values(), color=['red', 'yellow', 'green'])
    plt.title("Análisis de Sentimientos - Reseñas de Gift Cards")
    plt.xlabel("Categorías")
    plt.ylabel("Cantidad de Reseñas")
    
    # Generar nombre único para la imagen
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    image_name = f"result_img_{timestamp}.png"
    
    # Guardar en la carpeta results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
    print(f"Carpeta de resultados: {results_dir}")
    
    # Crear la carpeta si no existe
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            print("Carpeta 'results' creada.")
        except Exception as e:
            print(f"Error al crear la carpeta 'results': {e}")
            return
    
    # Ruta completa de la imagen
    image_path = os.path.join(results_dir, image_name)
    
    try:
        plt.savefig(image_path)  # Guardar la gráfica
        print(f"Gráfica guardada en: {image_path}")
    except Exception as e:
        print(f"Error al guardar la gráfica: {e}")
    
    plt.show()


# Punto de entrada del script
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print("Cargando reseñas...")
    reviews = load_reviews(os.path.join(BASE_DIR, "../data/Gift_Cards_reviews.jsonl"))

    if not reviews:
        print("No se encontraron reseñas para procesar. Revisa el archivo de datos.")
    else:
        print(f"Se cargaron {len(reviews)} reseñas.")

        print("Preprocesando reseñas...")
        reviews = [preprocess_text(review) for review in reviews]

        print("Analizando sentimientos...")
        sentiments = analyze_sentiments(reviews)

        print("Clasificando sentimientos...")
        categories = categorize_sentiments(sentiments)
        print(f"Resultados: {categories}")

        print("Mostrando gráfica...")
        plot_results(categories)
