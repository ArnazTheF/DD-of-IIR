import os
import pandas as pd
import numpy as np
import plotly.express as px
import logging
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


SENTENCES_FILE = None  # или путь к файлу, например: 'sentences.txt'
OUTPUT_DIR = '.'       # выходная директория

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Similarity():
    """
    Класс для вычисления косинусной близости между предложениями 
    с использованием различных методов эмбеддингов.
    """
    
    def __init__(self):
        """
        Инициализация класса Similarity.
        
        Загружает модели для эмбеддингов и устанавливает значения по умолчанию.
        """
        # Стандартные предложения для демонстрации
        self.sentences = ["Cosine similarity measures the angle between two vectors to determine their similarity",
                    "Deep learning is a subset of Machine Learning.",
                    "Machine learning models require large datasets to achieve high accuracy"
                ]
        self.output_dir = '.'
        
        try:
            # Загрузка модели трансформера для эмбеддингов предложений
            self.BidirectionalTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Ошибка загрузки модели трансформера: {e}")
            raise
        
        try:
            # Загрузка предварительно обученной модели Word2Vec
            self.word2vec = Word2Vec.load('word2vec.model')
        except Exception as e:
            logger.error(f"Ошибка загрузки модели Word2Vec: {e}")
            raise
        
    def w2v_sentence_embeddings(self, sentences):
        """
        Рассчитывает эмбеддинги предложений путем усреднения векторов слов
        с использованием модели Word2Vec.

        Parameters:
            sentences (list): Список предложений для обработки.

        Returns:
            numpy.ndarray: Массив эмбеддингов предложений.
        """
        sentence_embeddings = []
        
        for sentence in sentences:
            # Проверка типа входных данных
            if not isinstance(sentence, str):
                logger.warning(f"Пропущено нестроковое предложение: {sentence}")
                continue
                
            # Токенизация предложения на слова
            words = sentence.lower().split()
            
            # Получение векторов для каждого слова
            word_vectors = []
            for word in words:
                # Очистка слова от знаков препинания
                clean_word = ''.join(char for char in word if char.isalnum())
                if clean_word and clean_word in self.word2vec.wv:
                    try:
                        vector = self.word2vec.wv[clean_word]
                        word_vectors.append(vector)
                    except KeyError:
                        # Слово отсутствует в словаре модели
                        logger.debug(f"Слово '{clean_word}' не найдено в модели Word2Vec")
                    except Exception as e:
                        logger.warning(f"Ошибка получения вектора для слова '{clean_word}': {e}")
            
            # Усреднение векторов слов для получения эмбеддинга предложения
            if word_vectors:
                try:
                    sentence_embedding = np.mean(word_vectors, axis=0)
                except Exception as e:
                    logger.error(f"Ошибка усреднения векторов для предложения: {e}")
                    # Используем нулевой вектор в случае ошибки
                    sentence_embedding = np.zeros(self.word2vec.vector_size)
            else:
                # Если ни одного слова не найдено в модели, используем нулевой вектор
                logger.warning(f"Не найдено векторов для предложения: '{sentence}'")
                sentence_embedding = np.zeros(self.word2vec.vector_size)
                
            sentence_embeddings.append(sentence_embedding)
            
        return np.array(sentence_embeddings)
        
    def calculate(self, sentences=None, output_dir=None, method='transformer'):
        """
        Основной метод для вычисления матрицы косинусной близости.

        Parameters:
            sentences (str/list): Путь к файлу или список предложений.
            output_dir (str): Директория для сохранения результатов.
            method (str): Метод вычисления эмбеддингов ('transformer' или 'word2vec').

        Raises:
            NameError: Если файл с предложениями имеет неверное расширение.
            ValueError: При недопустимых значениях параметров.
            Exception: При ошибках обработки данных.
        """
        try:
            # Используем значения по умолчанию если параметры не указаны
            if sentences is None:
                sentences = self.sentences
            if output_dir is None:
                output_dir = self.output_dir
                
            # Чтение и валидация входных данных
            if isinstance(sentences, str): 
                if not sentences.endswith('.txt'):
                    raise NameError(f'File {sentences} should be txt file.')
                
                try:
                    with open(sentences, 'r', encoding='utf-8') as file:
                        sentences = file.readlines()
                        # Удаляем символы переноса строк
                        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
                except FileNotFoundError:
                    logger.error(f"Файл {sentences} не найден")
                    raise
                except Exception as e:
                    logger.error(f"Ошибка чтения файла {sentences}: {e}")
                    raise
                    
                if len(sentences) > 3:
                    raise ValueError(f'You should use only 3 sentences.')
            
            # Проверка что sentences является списком
            if not isinstance(sentences, list):
                raise ValueError("Sentences should be a list or path to a file")
                
            # Проверка что список предложений не пустой
            if len(sentences) == 0:
                raise ValueError("Sentences list cannot be empty")
            
            # Создание выходной директории
            try:
                os.makedirs(output_dir, exist_ok=True)
            except PermissionError:
                logger.error(f"Нет прав для создания директории {output_dir}")
                raise
            except Exception as e:
                logger.error(f"Ошибка создания директории {output_dir}: {e}")
                raise

            # Выбор метода для получения эмбеддингов
            if method == 'transformer':
                # Использование SentenceTransformer
                try:
                    embeddings = self.BidirectionalTransformer.encode(sentences)
                except Exception as e:
                    logger.error(f"Ошибка получения эмбеддингов трансформером: {e}")
                    raise
                file_suffix = '_transformer'
            elif method == 'word2vec':
                # Использование Word2Vec с усреднением векторов слов
                try:
                    embeddings = self.w2v_sentence_embeddings(sentences)
                except Exception as e:
                    logger.error(f"Ошибка получения эмбеддингов Word2Vec: {e}")
                    raise
                file_suffix = '_word2vec'
            else:
                raise ValueError("Method should be either 'transformer' or 'word2vec'")
            
            # Проверка корректности полученных эмбеддингов
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Не удалось получить эмбеддинги для предложений")
            
            # Расчет матрицы косинусной близости
            try:
                cosine_sim_matrix = cosine_similarity(embeddings)
            except Exception as e:
                logger.error(f"Ошибка вычисления косинусной близости: {e}")
                raise
            
            # Создание интерактивного графика с помощью Plotly
            labels = [f"Sentence {i+1}" for i in range(len(sentences))]
            try:
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
            except Exception as e:
                logger.error(f"Ошибка создания графика: {e}")
                raise
            
            # Настройка внешнего вида
            fig.update_layout(title_x=0.5)
            fig.update_xaxes(side="top")
            
            # Сохранение графика
            try:
                output_path = os.path.join(output_dir, f"sim_matrix{file_suffix}.html")
                fig.write_html(output_path)
                logger.info(f"График сохранен в {output_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения графика: {e}")
                raise
            
            # Логирование результатов
            logger.info(f"Cosine similarity matrix calculated using {method}:")
            for i in range(len(cosine_sim_matrix)):
                for j in range(len(cosine_sim_matrix[i])):
                    if i != j:  # Логируем только сравнения между разными предложениями
                        logger.info(f"Sentence {i+1} vs Sentence {j+1}: {cosine_sim_matrix[i][j]:.4f}")
                        
        except Exception as e:
            logger.error(f"Критическая ошибка в методе calculate: {e}")
            raise

def main():
    """
    Основная функция для запуска вычислений косинусной близости.
    """
    try:
        cos_sim = Similarity()
        
        # Расчет с использованием SentenceTransformer
        cos_sim.calculate(sentences=SENTENCES_FILE, output_dir=OUTPUT_DIR, method='transformer')
        
        # Расчет с использованием Word2Vec
        cos_sim.calculate(sentences=SENTENCES_FILE, output_dir=OUTPUT_DIR, method='word2vec')
        
        return True
    except Exception as e:
        logger.error(f"Ошибка в основной функции: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)