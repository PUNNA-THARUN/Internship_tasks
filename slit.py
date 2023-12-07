import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
import seaborn as sns
import re
from pdfminer.high_level import extract_text
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords


# Setting the page title as "Punna Project"
st.set_page_config(page_title= "Punna Project", page_icon="fire")

########################################################################
# Side bar with text 
st.sidebar.header("About Application : ", divider="red")

text = """This is a demo project to match 
the jobs and resumes.

○ NLP Project
○ By Punna Tharun
○ Instructed by Ashok sir.
○ Hyderabad
○ India

"""
st.sidebar.text(text)

if st.sidebar.button("LIKE", type="secondary"):
    st.sidebar.write("Thanks for your like 😊")
elif st.sidebar.button("DISLIKE", type="secondary"):
    st.sidebar.write("Please leave a comment to improve")
    st.sidebar.text_input("Comment here : ")
st.sidebar.subheader("",divider="red")

st.sidebar.link_button("GITHUB", "https://github.com/PUNNA-THARUN/", type="primary")
st.sidebar.link_button("LinkedIN", "https://www.linkedin.com/in/punna-tharun/", type="primary")

########################################################################

# title- used to add title of an app, Title is "Project"
st.title(":green[-------------------Project-------------------]")
st.info("😊______________________________Matching Jobs to Resume___________________________😊")
st.subheader("",divider="orange")
st.subheader("Upload your resume below to view matching jobs•••👇")

########################################################################

# Reading the CSV files using pandas
job_des = pd.read_csv("C:\\Users\\tharu\\OneDrive\\Desktop\\projets\\job_Data.csv")

# Dropping the unnecessary columns in job description
unnecessary_columns = ['job_ID', 'work_type', 'involvement', 'company_id', 'name', 'employees_count', 'total_applicants',
                       'linkedin_followers', 'details_id', 'industry', 'level', 'City', 'State']
job_des.drop(unnecessary_columns, axis=1, inplace=True)

#  we are doing text cleaning
def clean_job(JobText):
    JobText = re.sub('httpS+s*', ' ', JobText)  # remove URLs
    JobText = re.sub('RT|cc', ' ', JobText)  # remove RT and cc
    JobText = re.sub('#S+', '', JobText)  # remove hashtags
    JobText = re.sub('@S+', '  ', JobText)  # remove mentions
    JobText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', JobText)  # remove punctuations
    JobText = re.sub(r'[^\x00-\x7f]', r' ', JobText)
    JobText = re.sub(' +', ' ', JobText)  # remove extra whitespace
    JobText = re.sub(r'(?<!\w) +|\s+(?!\w)', ' ', JobText)
    JobText = re.sub(r'[0-9]', r' ', JobText)  # num removing
    return JobText

job_des['Cleaned_Job_Description'] = job_des.job_description.apply(lambda x: clean_job(x))
job_des['Cleaned_Job_Title'] = job_des.designation.apply(lambda x: clean_job(x))

job_text = " ".join(job_des['Cleaned_Job_Description'])
job_title_text = " ".join(job_des['Cleaned_Job_Title'])

tokenizer = nltk.WordPunctTokenizer()

# Tokenizing and processing job description
job_tokens = tokenizer.tokenize(job_text)
job_words = [job_word.lower() for job_word in job_tokens]
wn = WordNetLemmatizer()
lem_job_words = [wn.lemmatize(word) for word in job_words]
stop_words = stopwords.words('english')
new_job_words = [word for word in lem_job_words if word not in stop_words]

# Tokenizing and processing job title
job_title_tokens = tokenizer.tokenize(job_title_text)
job_title_words = [title_word.lower() for title_word in job_title_tokens]
lem_job_title_words = [wn.lemmatize(word) for word in job_title_words]
new_job_title_words = [word for word in lem_job_title_words if word not in stop_words]

# Reading the Resume
def read_pdf(file_path):
    return extract_text(file_path)

uploaded_file = st.file_uploader("PDF's only", type="pdf")

if uploaded_file is not None:
    st.write("Resume Uploaded successfully")
    pdf_text = read_pdf(uploaded_file)
    st.text_area("Your Resume", pdf_text)

    # Cleaning the text
    cleaned_text = clean_job(pdf_text)

    tokenizer = nltk.WordPunctTokenizer()
    tokens = tokenizer.tokenize(cleaned_text)

    # Lower casing words
    res_words = [res_word.lower() for res_word in tokens]

    # Lemmatizing the resume text
    lem_res_words = [wn.lemmatize(word) for word in res_words]

    # Stop words removal
    new_res_words = [word for word in lem_res_words if word not in stop_words]

    st.divider()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Gather resume
    resume_text = [" ".join(new_res_words)]
    job_description = [" ".join(new_job_words)]
    job_title = [" ".join(new_job_title_words)]

    # Matching
    if resume_text and job_description and job_title:
        # Vectorizing the documents
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(resume_text + job_description + job_title)

        # Calculating cosine similarity
        similarity_resume_description = cosine_similarity(vectors[0], vectors[1])[0][0]
        similarity_resume_title = cosine_similarity(vectors[0], vectors[2])[0][0]

        # Getting the top 3 matching job titles
        job_titles_similarity = {}
        for indx, title in enumerate(job_des["Cleaned_Job_Title"]):
            title_text = clean_job(title)
            title_tokens = tokenizer.tokenize(title_text.lower())
            lem_title_words = [wn.lemmatize(word) for word in title_tokens if word not in stop_words]
            title_words = " ".join(lem_title_words)
            vectors_title = vectorizer.transform([resume_text[0], title_words])
            similarity = cosine_similarity(vectors_title[0], vectors_title[1])[0][0]
            job_titles_similarity[title] = similarity

        top_3_jobs = sorted(job_titles_similarity.items(), key=lambda x: x[1], reverse=True)[:3]


        result_description = f"Matching Score : {similarity_resume_description * 100:.2f}%"

        st.header(result_description, divider="orange")

        st.subheader("Matched jobs:- ")
        # Display top 3 matching job titles
        st.text("Top 3 Matching Job Titles:")
        for i, (job_title, similarity) in enumerate(top_3_jobs, start=1):
            st.text(f"{i}. {job_title} - Matching Score: {similarity * 100:.2f}%")
        st.subheader("",divider="orange")
else:
    st.text("Please upload a resume")

########################################################################
# Reset button
if st.button("Reset",type="primary"):
        # Perform any reset action here
        st.experimental_rerun()

