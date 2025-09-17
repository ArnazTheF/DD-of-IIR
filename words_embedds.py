import traceback

import pandas as pd
import numpy as np
import nltk
import plotly.express as px
import logging

from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.data import LazyLoader

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILENAME = 'shakespeare-hamlet.txt'
VECTOR_SIZE = 50
WINDOW = 10
MIN_COUNT = 1
EPOCHS = 20

def download_nltk_resources():
    """
    Подкачиваете токенизатор и тексты, на одном из которых будем тренить w2v
    """
    resources = [
        ('gutenberg', 'corpora/gutenberg'),
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    for resource, path in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logger.error(f"Failed to download {resource}: {str(e)}")
            raise

def tokenizer(filename):
    """
    Предобработка предложений. По умолчанию Гамлет из предустановленного библиотекой nltk набора текстов, 
    однако можно использовать свой.
    filename - имя файла с текстом
    """
    try:
        hamlet = nltk.corpus.gutenberg.sents(filename) #Подгружаем текст в обработчик
        stop_words = set(stopwords.words('english')) #Язык можно менять
        tokens = [] #Инициализируем будущий массив токенов
        
        # В цикле обрабатываем каждое предложение, достаем из него токены и добавляем в массив
        for sentence in hamlet:
            try:
                tokens.append([word.lower() for word in sentence 
                             if word.isalpha() and word.lower() not in stop_words])
            except Exception as e:
                logger.warning(f"Error processing sentence: {sentence}. Error: {str(e)}")
                continue
                
        if not tokens:
            raise ValueError("No tokens generated from text")
            
        return tokens
        
    except Exception as e:
        logger.error(f"Error in tokenizer: {str(e)}")
        raise

def model_train(vector_size, window, min_count, epochs):
    """
    В данной функции мы инициализируем и тренируем модель на основе поданного на вход текста
    vector_size - размер вектора для каждого токена
    window - размер окна контекста для каждого токена
    epochs - число эпох для обучения
    mincount - токены встречающиеся меньше указанного тут числа раз НЕ СОХРАНЯЮТСЯ в векторном виде, 
    с ними надо будет работать чуть иначе
    """
    try:
        tokens = tokenizer(FILENAME) # Разиваем текст на токены
        
        if not tokens or all(len(sublist) == 0 for sublist in tokens):
            raise ValueError("No valid tokens for training")
        
        logger.info(f"Training model on {len(tokens)} sentences")
        
        # Инициализируем модель
        model = Word2Vec(
            sentences=tokens,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        model.train(tokens, total_examples=model.corpus_count, epochs=epochs) #Тренируем
        return model
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def save_model(model, filename="word2vec.model"):
    """
    Сохранение модели
    """
    try:
        model.save(filename)
        logger.info(f"Model successfully saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(filename="word2vec.model"):
    """
    Загрузка модели
    """
    try:
        model = Word2Vec.load(filename)
        logger.info(f"Model successfully loaded from {filename}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file {filename} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def embeddings_extractor(model):
    """
    model - w2v модель
    Функция для извлечения массива эмбеддингов токенов
    """
    try:
        if not model.wv:
            raise ValueError("Model doesn't contain word vectors")
        
        words = model.wv.index_to_key # Словарь модели
        if not words:
            raise ValueError("No words in model vocabulary")
        
        #Вот искомые вектора, сохраняйте их в нужном формате если будет необходимость
        word_vectors = np.array([model.wv[word] for word in words])
        return words, word_vectors
        
    except Exception as e:
        logger.error(f"Error extracting embeddings: {str(e)}")
        raise

def eval_mode(model, word='england'):
    """
    model - w2v модель
    Функция для проверки работы модели
    """
    try:
        if word not in model.wv:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        word_vector = np.array(model.wv[word]) #Достаем из модели вектора
        logger.info(f"Vector for '{word}': {word_vector}")
        logger.info(f"Vector size: {len(word_vector)}")
        return word_vector
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise

def embeddings_tsne(words, word_vectors, output_file="tsne_word2vec_words_v2.html"):
    """
    TSNE, если интересна визуализация векторов слов на плоскости
    words - словарь обработанных моделью слов
    word_vectors - массив эмбеддингов слов
    """
    try:
        if len(words) != len(word_vectors):
            raise ValueError("Words and vectors length mismatch")
        
        if len(word_vectors) < 2:
            raise ValueError("Not enough vectors for t-SNE")
        
        tsne_model = TSNE(n_components=2, random_state=2025, perplexity=30) #Загружаем модель TSNE
        #Запускаем алгоритм TSNE, которые проецирует n-размерные эмбеддинги в двумерное или трехмерное представление в зависимости от настроек
        reduced_vectors = tsne_model.fit_transform(word_vectors)

        # Создаем датафрейм с помощью которого будем визуализировать вектора
        df_tsne = pd.DataFrame({
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1],
            'word': words
        })
        
        # Сама визуализация
        fig = px.scatter(
            df_tsne,
            x='x',
            y='y',
            hover_name='word',
            title='t-SNE визуализация векторов слов (Word2Vec)',
            width=1000,
            height=1000
        )
        
        # plotly использует web-интерфейс для визуализации, поэтому загружаем график в html-файл
        fig.write_html(output_file)
        logger.info(f"t-SNE visualization saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in t-SNE visualization: {str(e)}")
        raise

def main():
    try:
        # Загрузка ресурсов
        download_nltk_resources()
        
        # Обучение или загрузка модели
        try:
            model = load_model()
        except (FileNotFoundError, Exception):
            logger.info("Training new model...")
            model = model_train(VECTOR_SIZE, WINDOW, MIN_COUNT, EPOCHS)
            #save_model(model)
        
        # Извлечение и оценка эмбеддингов
        words, word_embedds = embeddings_extractor(model)
        eval_mode(model)
        
        # Визуализация
        # embeddings_tsne(words, word_embedds)
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)