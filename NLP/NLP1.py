##################################################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##################################################

##################################################
# Business Problem
##################################################
# Amazon üzerinden satışlarını gerçekleştiren ev tesktili ve günlük giyim odaklı üretimler yapan Kozmos ürünlerine
# gelen yorumları analiz ederek ve aldığı şikayetlere göre özelliklerini geliştirerek satışlarını artırmayı hedeflemektedir.
# Bu hedef doğrultusunda yorumlara duygu analizi yapılarak etiketlencek ve   etiketlenen veri ile sınıflandırma modeli
# oluşturulacaktır.

##################################################
# Veri Seti Hikayesi
##################################################
# Veri seti belirli bir ürün grubuna ait yapılan yorumları, yorum başlığını, yıldız sayısını ve yapılan yorumu
# kaç kişinin faydalı bulduğunu belirten değişkenlerden oluşmaktadır.

# Review: Ürüne yapılan yorum
# Title: Yorum içeriğine verilen başlık, kısa yorum
# HelpFul: Yorumu faydalı bulan kişi sayısı
# Star: Ürüne verilen yıldız sayısı

##############################################################
# Görevler
##############################################################
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings
import matplotlib

matplotlib.use("Qt5Agg")
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# Görev 1: Metin ön işleme işlemleri.
# 1. amazon.xlsx datasını okutunuz.
df = pd.read_excel("datasets/amazon.xlsx")
df.head()

# 2. "Review" değişkeni üzerinde
# a. Tüm harfleri küçük harfe çeviriniz
df["Review"] = df["Review"].str.lower()

# b. Noktalama işaretlerini çıkarınız
df["Review"] = df["Review"].str.replace('[^\w\s]', '')

# c. Yorumlarda bulunan sayısal ifadeleri çıkarınız
df["Review"] = df["Review"].str.replace('\d', '')

# d. Bilgi içermeyen kelimeleri (stopwords) veriden çıkarınız
import nltk
# nltk.download('stopwords')
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# e. 1000'den az geçen kelimeleri veriden çıkarınız
temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()
drops = temp_df[temp_df <= 1000]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# f. Lemmatization işlemini uygulayınız
#Bu kod örneğin, yorumlardaki kelimelerin farklı formlarını aynı temel forma dönüştürerek daha tutarlı bir analiz yapılmasına olanak tanır.
# Örneğin, "running" ve "runs" kelimelerini aynı temel kelime olan "run" olarak ele alabilir.
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# Görev 2: Metin Görselleştirme
# Adım 1: Barplot görselleştirme işlemi
# a. "Review" değişkeninin içerdiği kelimeleri frekanslarını hesaplayınız, tf olarak kaydediniz
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

# b. tf dataframe'inin sütunlarını yeniden adlandırınız: "words", "tf" şeklinde
tf.columns = ["words", "tf"]

# c. "tf" değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirme işlemini tamamlayınız.
tf[tf["tf"] > 500].plot.barh(x="words", y="tf")
plt.show()

# Adım 2: WordCloud görselleştirme işlemi
text = " ".join(i for i in df.Review)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Görev 3: Duygu Analizi
# Adım 1: Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturunuz
sia = SentimentIntensityAnalyzer()

# Adım 2: SentimentIntensityAnalyzer nesnesi ile polarite puanlarının incelenmesi
# a. "Review" değişkeninin ilk 10 gözlemi için polarity_scores() hesaplayınız
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

# b. İncelenen ilk 10 gözlem için compund skorlarına göre filtrelenerek tekrar gözlemleyiniz
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])
df["polarity_score"] = df["Review"].apply(lambda x: sia.polarity_scores(x)["compound"])

# c. 10 gözlem için compound skorları 0'dan büyükse "pos" değilse "neg" şeklinde güncelleyiniz
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# d. "Review" değişkenindeki tüm gözlemler için pos-neg atamasını yaparak yeni bir değişken olarak dataframe'e ekleyiniz
df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# NOT:SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorum sınıflandırma makine öğrenmesi modeli için bağımlı değişken oluşturulmuş oldu.


# Görev 4: Makine öğrenmesine hazırlık!
# Adım 1: Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train test olara ayırınız.
# Test-Train
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

# Adım 2: Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte.
# a. TfidfVectorizer kullanarak bir nesne oluşturunuz.
# b. Daha önce ayırmış olduğumuz train datamızı kullanarak oluşturduğumuz nesneye fit ediniz.
# c. Oluşturmuş olduğumuz vektörü train ve test datalarına transform işlemini uygulayıp kaydediniz.

# TF-IDF Word Level
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)


###############################
# Görev 6: Modelleme (Random Forest)
###############################
# Adım 1: Random Forest modeliiletahminsonuçlarınıngözlenmesi;
         # a. RandomForestClassifier modelini kurup fit ediniz.
         # b. cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()



random_review = pd.Series(df["Review"].sample(1).values)
yeni_yorum = CountVectorizer().fit(train_x).transform(random_review)
pred = rf_model.predict(yeni_yorum)
print(f'Review:  {random_review[0]} \n Prediction: {pred}')



############################################################################################################################