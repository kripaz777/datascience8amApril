from fastapi import FastAPI
import pickle
import pandas as pd
app = FastAPI()
#loading model
dbfile = open('LogisticRegression.pickle', 'rb')
model1 = pickle.load(dbfile)

dbfile = open('LinearRegression.pickle', 'rb')
model2 = pickle.load(dbfile)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/news/{data}")
def news_classification(data:str):
    data = data.replace("%20"," ")
    d = {'news':[data]}
    df = pd.DataFrame(d)
    result = model1.predict(df['news'])[0]
    # print(df['news'])
    return {'output':result}

@app.post("/salary")
def salary_classification(age:int,edu:str,exp:float):
   
    if edu == "Bachelor":
        b = 1; m =0; p =0
    elif edu == "Master":
        b = 0; m = 1; p = 0
    else:
        b = 0; m = 0; p = 1

    df = pd.DataFrame(
        {'Age':[age],
        "Years of Experience":[exp],
        "Bachelor's":[b],
        "Master's":[m],
        "PhD":[p]}
    )
    result = int(model2.predict(df)[0])
    print(result)
    return {'output':result}