import streamlit as st
import requests
from io import BytesIO
import pandas as pd

# Specify the API endpoint
api_url = "http://127.0.0.1:8000/prediction"

# Streamlit UI
st.title("CV Prediction App")

# Set the overall layout width
st.markdown(
    """
    <style>
        .dataframe {
            max-width: 3000px;
            margin: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Number of Jobs Input
number_of_jobs = st.number_input("Number of Jobs", min_value=1, step=1, max_value=3621)

# Make prediction on button click
if st.button("Make Prediction"):
    if uploaded_file is not None:
        # Prepare the data to be sent in the request
        files = {"cv": ("file.pdf", uploaded_file)}
        params = {"number_of_jobs": number_of_jobs}

        # Send the POST request with parameters
        response = requests.post(api_url, files=files, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            prediction_data = response.json()["prediction"]

            data_df = pd.DataFrame(prediction_data)

            # Display the DataFrame with st.data_editor
            st.data_editor(
                data_df,
                column_config={
                    "job_link": st.column_config.LinkColumn(
                        "Job Link",
                        help="The top WebSite",
                        validate="^https://[a-z]+\.streamlit\.app$",
                        max_chars=100,
                    ),
                },
                hide_index=True,
            )
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("Please upload a PDF file.")
