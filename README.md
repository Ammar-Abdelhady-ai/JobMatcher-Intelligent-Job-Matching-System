# JobMatcher-Intelligent-Job-Matching-System

JobMatcher automates job search by scraping data from LinkedIn, Indeed, Bayt, and Wuzzuf. It leverages transformer models to summarize user CVs and calculates cosine similarity, presenting personalized job recommendations for seamless career opportunities.

## Job Matcher

Job Matcher is a Python project designed to help users find suitable job opportunities by scraping data from multiple job websites such as LinkedIn, Indeed, Bayt, and Wuzzuf. The project performs data scraping, preprocessing, and feature engineering, consolidating the information into a single CSV file. Users can input their CV, which is then summarized using transformer models. The summarized data is used to calculate cosine similarity against job data, helping users discover the most relevant job opportunities.

## Features

- **Data Scraping:** Collect job data from LinkedIn, Indeed, Bayt, and Wuzzuf.
- **Data Processing:** Apply preprocessing techniques and feature engineering for improved data quality.
- **User CV Input:** Summarize user CVs using transformer models.
- **Job Matching:** Calculate cosine similarity to identify the most relevant job opportunities.

## Deploying with FastAPI

1. **Run the FastAPI application:**

    ```bash
    uvicorn fastapi_app:app --reload
    ```

## Connecting with Streamlit App

2. **Run the Streamlit app:**

    ```bash
    streamlit run streamlit_app.py
    ```

## Requirements

- **Python 3.x:** The project is built using Python 3.x.
- **requirements.txt:** Install project dependencies by running the following command:

    ```bash
    pip install -r requirements.txt
    ```
Please replace your existing README content with this updated version. Save the changes, and your README file should now be formatted correctly.
