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


#Подкачиваете токенизатор и мини-библу с текстами на одном из которых будем тренить w2v
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

#Предобработка предложений. Я брал Гамлета, но ничего не мешает взять свой текст
nltk.corpus.gutenberg.fileids()
hamlet = nltk.corpus.gutenberg.sents('shakespeare-hamlet.txt')
stop_words = set(stopwords.words('english')) #Язык можно менять
tokens = []
for sentence in hamlet:
    tokens.append([word.lower() for word in sentence if word.isalpha() and word not in stop_words])
#print(tokens)

"""
mincount - крч слова встречающиеся меньше указанного тут числа раз НЕ СОХРАНЯЮТСЯ в векторном виде, 
с ними надо будет работать чуть иначе, но это уже не моя проблема, хехехе
"""
model = Word2Vec(tokens, vector_size=50, window=10, min_count=1, workers=4) 
#Либо так если очень хочется
#model.build_vocab(tokens) 
model.train(tokens, total_examples=model.corpus_count, epochs=20)

#Тестим:
try:
    similarity = model.wv.similarity('hamlet', 'prince')
    print(f"Similarity: {similarity:.4f}")
except KeyError as e:
    print(f"KeyError: {e}")

    
# model.save("word2vec.model") - Так если надо модель можно сохранить
# model = Word2Vec.load("word2vec.model") - Так модель можно загрузить

words = model.wv.index_to_key
word_vectors = np.array([model.wv[word] for word in words]) #Вот искомые вектора, сохраняйте их в нужном формате если будет необходимость
#print(word_vectors)

#Для конкретных слов:
word_vector = np.array(model.wv['england'])
print(f"Vector for your word: {word_vector}")
print(f"It's size: {len(word_vector)}")

"""
#Тут красивое TSNE, но это только если вам очень интересно

tsne_model = TSNE(n_components=2, random_state=2025, perplexity=25)
reduced_vectors = tsne_model.fit_transform(word_vectors)

df_tsne = pd.DataFrame({
    'x': reduced_vectors[:, 0],
    'y': reduced_vectors[:, 1],
    'word': words
})

fig = px.scatter(
    df_tsne,
    x='x',
    y='y',
    hover_name='word',
    title='t-SNE визуализация векторов слов (Word2Vec)',
    width=1000,
    height=1000
)

fig.write_html("tsne_word2vec_words_v2.html")
"""
