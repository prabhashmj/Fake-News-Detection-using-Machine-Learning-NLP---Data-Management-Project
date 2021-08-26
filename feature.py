import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf


def preprocessing (query):
    import csv

    with open('tempfile.csv', 'w', newline='') as f:
        fieldnames = ['text']
        thewriter = csv.DictWriter(f, fieldnames = fieldnames)

        thewriter.writeheader()
        thewriter.writerow({'text': query})


    true = pd.read_csv("G:/Academic/SEM 07/Data Management project/App/input/True.csv")
    false = pd.read_csv("G:/Academic/SEM 07/Data Management project/App/input/Fake.csv")
    true['category'] = 1
    false['category'] = 0
    df = pd.concat([true,false]) #Merging the 2 datasets
    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']

    filedata = pd.read_csv("./tempfile.csv")
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    #Removing the square brackets
    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)
    # Removing URL's
    def remove_between_square_brackets(text):
        return re.sub(r'http\S+', '', text)
    #Removing the stopwords from text
    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)
    #Removing the noisy text
    def denoise_text(text):
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        text = remove_stopwords(text)
        return text
    #Apply function on review column
    filedata['text']=filedata['text'].apply(denoise_text)

    def get_corpus(text):
        words = []
        for i in text:
            for j in i.split():
                words.append(j.strip())
        return words
    corpus = get_corpus(filedata.text)

    from collections import Counter
    counter = Counter(corpus)
    most_common = counter.most_common(10)
    most_common = dict(most_common)

    from sklearn.feature_extraction.text import CountVectorizer
    def get_top_text_ngrams(corpus, n, g):
        vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]

    x_train,x_test,y_train,y_test = train_test_split(df.text,df.category,test_size = 0.3, random_state = 0)

    x_filedata = filedata.text

    max_features = 10000
    maxlen = 300

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train)
    tokenized_train = tokenizer.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

    tokenized_test = tokenizer.texts_to_sequences(x_filedata)
    X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)



    X_test = X_test.tolist()

    print("PRINTING IN FEATURE:")
    print(X_test)
    return X_test