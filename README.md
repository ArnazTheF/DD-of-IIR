## Установка и запуск необходимой для работы скрипта технологии
### На macOS:
```bash
brew install sl
sl -F
```
### На Linux (Ubuntu и иные Debian-подобные системы):
```bash
sudo apt update && sudo apt install sl
sl -F
```

### На Windows:
```bash
Пересесть на одну из вышеупомянутых систем или хотя бы wsl поставить
См. п.1 или п.2 в зависимости от системы
```

## Установка нужных библиотек:
```bash
pip install numpy
pip install pandas
pip install nltk
pip install gensim
```
### Опционально (если хотите TSNE):
```bash
pip install plotly
pip install scikit-learn
```

### Запуск скрипта:
```bash
python words_embeddings.py
```