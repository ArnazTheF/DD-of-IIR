import pandas as pd
import numpy as np
import nltk
import plotly.express as px

from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import (
    word_tokenize,
    TweetTokenizer,
    WordPunctTokenizer,
    WhitespaceTokenizer,
    LegalitySyllableTokenizer,
    SyllableTokenizer,
)

#Подкачиваете токенизатор и тексты, на одном из которых будем тренить w2v
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


FILENAME = 'shakespeare-hamlet.txt'
VECTOR_SIZE = 50
WINDOW = 10
MIN_COUNT = 1
EPOCHS = 20


def tokenizer(filename):
    """
    Предобработка предложений. По умолчанию Гамлет из предустановленного библиотекой nltk набора текстов, 
    однако можно использовать свой.
    filename - имя файла с текстом
    """
    hamlet = nltk.corpus.gutenberg.sents(filename) #Подгружаем текст в обработчик
    stop_words = set(stopwords.words('english')) #Язык можно менять
    tokens = [] #Инициализируем будущий массив токенов
    # В цикле обрабатываем каждое предложение, достаем из него токены и добавляем в массив
    for sentence in hamlet: 
        tokens.append([word.lower() for word in sentence if word.isalpha() and word not in stop_words])
    return tokens

def model_train(vector_size, window, min_count, epochs):
    """
    В данной функции мы инициализируем и тренируем модель на основе поданного на вход текста
    vector_size - размер вектора для каждого токена
    window - размер окна контекста для каждого токена
    epochs - число эпох для обучения
    mincount - токены встречающиеся меньше указанного тут числа раз НЕ СОХРАНЯЮТСЯ в векторном виде, 
    с ними надо будет работать чуть иначе
    """
    tokens = tokenizer(FILENAME) # Разиваем текст на токены
    model = Word2Vec(tokens, vector_size=vector_size, window=window, min_count=min_count, workers=4) #Инициализируем модель
    model.train(tokens, total_examples=model.corpus_count, epochs=epochs) #Тренируем
    return model

def save_model(model):
    model.save("word2vec.model") #Так если надо модель можно сохранить
    return None

def load_model():
    model = Word2Vec.load("word2vec.model") #Так модель можно загрузить
    return model

def embeddings_extractor(model):
    """
    model - w2v модель
    Функция для извлечения массива эмбеддингов токенов
    """
    words = model.wv.index_to_key # Словарь модели
    word_vectors = np.array([model.wv[word] for word in words]) #Вот искомые вектора, сохраняйте их в нужном формате если будет необходимость
    return words, word_vectors

def eval_mode(model):
    """
    model - w2v модель
    Функция для проверки работы модели
    """
    word_vector = np.array(model.wv['england']) #Достаем из модели вектора
    print(f"Vector for your word: {word_vector}")
    print(f"It's size: {len(word_vector)}")

def embeddings_tsne(words, word_vectors):
    """
    TSNE, если интересна визуализация векторов слов на плоскости
    words - словарь обработанных моделью слов
    word_vectors - массив эмбеддингов слов
    """
    tsne_model = TSNE(n_components=2, random_state=2025, perplexity=25) #Загружаем модель
    reduced_vectors = tsne_model.fit_transform(word_vectors) #Запускаем алгоритм TSNE, которые проецирует n-размерные эмбеддинги в двумерное или трехмерное представление в зависимости от настроек

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
    fig.write_html("tsne_word2vec_words_v2.html")

model = model_train(VECTOR_SIZE, WINDOW, MIN_COUNT, EPOCHS)
words, word_embedds = embeddings_extractor(model)
eval_mode(model)
#embeddings_tsne(words, word_embedds)
"""

"""
