from fastapi import FastAPI
import pandas as pd
import nltk
import re
import pickle 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()

with open('LogisticRegression.pickle', 'rb') as file:
    model = pickle.load(file)
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.get("/news/{news_data}")
def read_news(news_data: str):
    cat = "News Category"
    df = pd.DataFrame(
        {'news':[news_data]}
    )
    df['news'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), df['news']))
    df['news'] = df['news'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

    predict_news_cat = model.predict(df['news'])

    return {"predicted_category": predict_news_cat[0]}