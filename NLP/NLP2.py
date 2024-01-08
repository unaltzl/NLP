from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Qt5Agg")
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


df = pd.read_csv("datasets/wiki_data.csv")
df.head()

#Metin ön işleme için clean_text adında fonksiyon oluşturunuz. Fonksiyon;
#Büyük küçük harf dönüşümü,
#Noktalama işaretlerini çıkarma,
#Numerik ifadeleri çıkarma Işlemlerini gerçekleştirmeli

def clean_text(dataframe):
    dataframe["text"] = dataframe["text"].str.lower()
    dataframe["text"] = dataframe["text"].str.replace('[^\w\s]', '')
    dataframe["text"] = dataframe["text"].str.replace('\d', '')
    return df
clean_text(df)

#Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri çıkaracak remove_stopwords adında fonksiyon yazınız.
def  remove_stopwords(dataframe):
    sw = stopwords.words('english')
    dataframe["text"] = dataframe["text"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    return df
remove_stopwords(df)

#Metinde az geçen (1500'den az) kelimeleri bulunuz. Ve bu kelimeleri metin içerisinden çıkartınız.
temp_df = pd.Series(' '.join(df['text']).split()).value_counts()
drops = temp_df[temp_df <= 1500]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Lemmatization işlemi yapınız.
df["text"].apply(lambda x: TextBlob(x).words).head()
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df['text'].head(10)

# Lemmatization işlemi yapınız.
tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

#: Bir önceki adımda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.
tf[tf["tf"] > 7500].plot.barh(x="words", y="tf")
plt.show()

#Kelimeleri WordCloud ile görselleştiriniz.
text = " ".join(i for i in df.text)
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#################
#FONKSİYONLAŞTIRMA
#################
def data_prep(dataframe, barplot=False, wordcloud=False):
    def clean_text(dataframe):
        dataframe["text"] = dataframe["text"].str.lower()
        dataframe["text"] = dataframe["text"].str.replace('[^\w\s]', '')
        dataframe["text"] = dataframe["text"].str.replace('\d', '')
        return dataframe  # Corrected from df to dataframe

    clean_text(dataframe)

    def remove_stopwords(dataframe):
        sw = stopwords.words('english')
        dataframe["text"] = dataframe["text"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
        return dataframe  # Corrected from df to dataframe

    remove_stopwords(dataframe)

    # rare_words
    temp_df = pd.Series(' '.join(dataframe['text']).split()).value_counts()
    drops = temp_df[temp_df <= 1500]
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

    # lemmatization
    dataframe["text"].apply(lambda x: TextBlob(x).words).head()
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # frequency
    tf = dataframe["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["words", "tf"]
    tf.sort_values("tf", ascending=False)

    # barplot
    if barplot:
        tf[tf["tf"] > 7500].plot.barh(x="words", y="tf")
        plt.show(block=True)

    # wordcloud
    if wordcloud:
        text = " ".join(i for i in dataframe.text)
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show(block=True)

    return dataframe

# Example usage
data_prep(df, barplot=False, wordcloud=True)















