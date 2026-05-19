import streamlit as st
import pandas as pd
import pickle

st.title('Salary Prediction System!')
st.image("https://cdn.slidesharecdn.com/ss_thumbnails/salarypredictionpptfinal15-4-24-240417144038-8b26979a-thumbnail.jpg?width=640&height=640&fit=bounds")

#loading model
dbfile = open('LinearRegression.pickle', 'rb')
model = pickle.load(dbfile)

age = st.number_input("Enter age = ",18,60)
exp = st.number_input("Enter Years of Experience = ", 0, 30)
edu = st.selectbox("Education = ", ["Bachelor's","Master's","PhD"])

if edu == "Bachelor's":
    b = 1; m =0; p =0
elif edu == "Master's":
    b = 0; m = 1; p = 0
else:
    b = 0; m = 0; p = 1

if st.button("Submit"):
    #Age	Years of Experience	Salary	Bachelor's	Master's	PhD
    df = pd.DataFrame(
        {'Age':[age],
         "Years of Experience":[exp],
         "Bachelor's":[b],
         "Master's":[m],
         "PhD":[p]}
    )
    result = model.predict(df)
    st.dataframe(df)
    st.write(result)