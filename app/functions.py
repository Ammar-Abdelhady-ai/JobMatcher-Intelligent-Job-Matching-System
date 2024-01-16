import os
import tempfile
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import numpy as np



def extract_text_from_pdf(pdf_content):
    text = ''
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_content)
        temp_path = temp_file.name

        pdf_document = fitz.open(temp_path)
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text += page.get_text()

    pdf_document.close()  # Close the PDF document explicitly
    os.remove(temp_path)  # Remove the temporary file after use
    return str(text.replace("\xa0", ""))


def get_most_similar_job(data, cv_vect, df_vect):
    for i in range(0, len([data])):
        distances = cosine_similarity(cv_vect[i], df_vect).flatten()
        indices = np.argsort(distances)[::-1]

    return indices
