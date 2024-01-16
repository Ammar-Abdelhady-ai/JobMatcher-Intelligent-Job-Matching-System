import threading
from functions import extract_text_from_pdf, get_most_similar_job
from fastapi import  UploadFile, HTTPException, FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


summarizer = ""
def define_summarizer():
    from transformers import pipeline
    global summarizer
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("\n\n definition Done")
define = threading.Thread(target=define_summarizer)
define.start()

def fit_threads(text):
    define.join()

    ######## Handel Sumarization model

    a = threading.Thread(target=summarization, args=(text[0],))
    b = threading.Thread(target=summarization, args=(text[1],))
    c = threading.Thread(target=summarization, args=(text[-1],))

    # Start all threads
    a.start()
    b.start()
    c.start()

    # Wait for all threads to finish
    a.join()
    b.join()
    c.join()
    print("Summarization Done")



df = pd.read_csv("all.csv")
concatenated_column = pd.concat([df['job_title'] + df['job_description'] + df['job_requirements'], df['city_name']], axis=1).astype(str).agg(''.join, axis=1)
x = concatenated_column
y = df["label"]
vectorizer = TfidfVectorizer(stop_words='english')

vectorizer.fit(x)
df_vect = vectorizer.transform(x)
print(df.shape, len(df))
# Initialize the summarizer model



######### using summarizer model
summ_data = []

def summarization(text):
    global summ_data
    part = summarizer(text, max_length=150, min_length=30, do_sample=False)
    summ_data.append(part[0]["summary_text"].replace("\xa0", ""))


app = FastAPI(project_name="cv")

@app.get("/")
async def read_root():
    return {"Hello": "World, Project name is : CV Description"}

@app.post("/prediction")
async def detect(cv: UploadFile, number_of_jobs: int):
    
    if (type(number_of_jobs) != int) or (number_of_jobs < 1) or (number_of_jobs > df.shape[0]):
        raise HTTPException(
            status_code=415, detail = f"Please enter the number of jobs you want as an ' integer from 1 to {int(df.shape[0]) - 1} '."
        )
    
    if cv.filename.split(".")[-1] not in ("pdf") :
        raise HTTPException(
            status_code=415, detail="Please inter PDF file "
        )



    cv_data = extract_text_from_pdf(await cv.read())
    index = len(cv_data)//3
    text = [cv_data[:index], cv_data[index:2*index], cv_data[2*index:]]
    fit_threads(text)
    
    data = " .".join(summ_data)
    summ_data.clear()
    cv_vect = vectorizer.transform([data])
    indices = get_most_similar_job(data=data, cv_vect=cv_vect, df_vect=df_vect)
    # Check if all threads have finished
    print("ALL Done \n\n")
    
    prediction_data = df.iloc[indices[:number_of_jobs]].applymap(lambda x: str(x)).to_dict(orient='records')
    


    return {"prediction": prediction_data}
