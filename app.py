import streamlit as st
import pandas as pd
import nltk
import re
import pickle 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
st.title('Text Classification!')

nltk.download('stopwords')
words = stopwords.words("english")
stemmer = PorterStemmer()

with open('LogisticRegression.pickle', 'rb') as file:
    model = pickle.load(file)

data = st.text_area("Enter news = ")
if st.button('Submit'):
    df = pd.DataFrame(
        {'news':[data]}
    )
    st.dataframe(df)
    df['news'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), df['news']))
    df['news'] = df['news'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

    predict_news_cat = model.predict(df['news'])
    st.success(f"The news category is = {predict_news_cat[0]}")
    