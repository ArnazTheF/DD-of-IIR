import os
import pandas as pd
import numpy as np
import plotly.express as px
import logging
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Константы вместо аргументов командной строки
SENTENCES_FILE = None  # или путь к файлу, например: 'sentences.txt'
OUTPUT_DIR = '.'       # выходная директория

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Similarity():
    def __init__(self):
        self.sentences = ["Cosine similarity measures the angle between two vectors to determine their similarity",
                    "The cosine similarity metric evaluates how close two vectors are by calculating the cosine of the angle between them",
                    "Machine learning models require large datasets to achieve high accuracy"
                ]
        self.output_dir = '.'
        self.BidirectionalTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.word2vec = Word2Vec.load('word2vec.model')
        
    def w2v_sentence_embeddings(self, sentences):
        """
        Рассчитывает эмбеддинги предложений путем усреднения векторов слов
        с использованием модели Word2Vec
        """
        sentence_embeddings = []
        
        for sentence in sentences:
            # Токенизация предложения на слова
            words = sentence.lower().split()
            
            # Получение векторов для каждого слова
            word_vectors = []
            for word in words:
                # Очистка слова от знаков препинания
                clean_word = ''.join(char for char in word if char.isalnum())
                if clean_word and clean_word in self.word2vec.wv:
                    vector = self.word2vec.wv[clean_word]
                    word_vectors.append(vector)
            
            # Усреднение векторов слов для получения эмбеддинга предложения
            if word_vectors:
                sentence_embedding = np.mean(word_vectors, axis=0)
            else:
                # Если ни одного слова не найдено в модели, используем нулевой вектор
                sentence_embedding = np.zeros(self.word2vec.vector_size)
                
            sentence_embeddings.append(sentence_embedding)
            
        return np.array(sentence_embeddings)
        
    def calculate(self, sentences=None, output_dir=None, method='transformer'):
        if sentences is None:
            sentences = self.sentences
        if output_dir is None:
            output_dir = self.output_dir
            
        # Чтение и валидация входных данных
        if isinstance(sentences, str): 
            if not sentences.endswith('.txt'):
                raise NameError(f'File {sentences} should be txt file.')
            with open(sentences, 'r') as file:
                sentences = file.readlines()
                if len(sentences) > 3:
                    raise ValueError(f'You should use only 3 sentences.')
        os.makedirs(output_dir, exist_ok=True)

        # Выбор метода для получения эмбеддингов
        if method == 'transformer':
            # Использование SentenceTransformer
            embeddings = self.BidirectionalTransformer.encode(sentences)
            file_suffix = '_transformer'
        elif method == 'word2vec':
            # Использование Word2Vec с усреднением векторов слов
            embeddings = self.w2v_sentence_embeddings(sentences)
            file_suffix = '_word2vec'
        else:
            raise ValueError("Method should be either 'transformer' or 'word2vec'")
        
        # Расчет матрицы косинусной близости
        cosine_sim_matrix = cosine_similarity(embeddings)
        
        # Создание интерактивного графика с помощью Plotly
        labels = [f"Sentence {i+1}" for i in range(len(sentences))]
        fig = px.imshow(
            cosine_sim_matrix,
            text_auto=True,
            color_continuous_scale='YlGnBu',
            range_color=[-1, 1],
            labels=dict(x="", y="", color="Cosine Similarity"),
            x=labels,
            y=labels,
            title=f"Cosine Similarity Matrix for Sentence Embeddings ({method})"
        )
        
        # Настройка внешнего вида
        fig.update_layout(title_x=0.5)
        fig.update_xaxes(side="top")
        
        # Сохранение графика
        fig.write_html(os.path.join(output_dir, f"sim_matrix{file_suffix}.html"))
        
        # Логирование результатов
        logger.info(f"Cosine similarity matrix calculated using {method}:")
        for i in range(len(cosine_sim_matrix)):
            for j in range(len(cosine_sim_matrix[i])):
                if i != j:  # Логируем только сравнения между разными предложениями
                    logger.info(f"Sentence {i+1} vs Sentence {j+1}: {cosine_sim_matrix[i][j]:.4f}")

def main():
    cos_sim = Similarity()
    
    # Расчет с использованием SentenceTransformer
    cos_sim.calculate(sentences=SENTENCES_FILE, output_dir=OUTPUT_DIR, method='transformer')
    
    # Расчет с использованием Word2Vec
    cos_sim.calculate(sentences=SENTENCES_FILE, output_dir=OUTPUT_DIR, method='word2vec')
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)