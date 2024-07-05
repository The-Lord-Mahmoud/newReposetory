import pandas as pd
from os import replace
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#from google.colab import drive
#from google.colab import files
#drive.mount('/content/drive')


df = pd.read_csv("/content/spotify_millsongdata.csv", sep=",")

df = df.sample(2000).drop('link', axis=1).reset_index(drop=True)
df['text']=df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex=True)


stm = PorterStemmer()

def token(text):
  tokens = nltk.word_tokenize(text)
  return " ".join([stm.stem(token) for token in tokens])


df['text'].apply(lambda x: token(x))

tfidf = TfidfVectorizer(analyzer = 'word',stop_words='english')

mat = tfidf.fit_transform(df['text'])

cos_similar = cosine_similarity(mat)

def recommendation(songName):
  index = df[df['song'] == songName].index[0]
  dis=sorted(list(enumerate(cos_similar[index])),reverse=True,key=lambda x:x[1])
  rec_songs = []
  for i in dis[1:20]:
    rec_songs.append(df.iloc[i[0]].song)
  return rec_songs

recommendation("Piano")