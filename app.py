# setup
# first load is slow - can we cache the nltk downloads?
import streamlit as st
import altair as alt
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import pickle

# dev
from vega_datasets import data


def main():
    st.sidebar.title('Job Atlas')
    # st.sidebar.markdown("You must unlearn what you have learned…No! Try not. Do. Or do not. There is no try.")
    dashboards = ['Where are you now?','Where do you want to go?']
    dashboard = st.sidebar.selectbox("What do you want to do?", dashboards, index=0)

    if dashboard == 'Where are you now?':

        st.title('Find yourself on the job map')

        #dev add max characters
        user_title = st.text_input("Tell us your job title", value='Data Scientist')
        user_jd = st.text_input(
            "Give us a short description of what you do",
            value = 'Use data to support and automate business decisions',
            max_chars=600
            )
        user_input = user_title + ' ' + user_jd
        plot_data = pd.read_csv('./static/job_embeddings.csv')
        reducer = pickle.load(open('./static/reducer.sav', 'rb'))
        tfidf_vectorizer = pickle.load(open('./static/tfidf_vectorizer.sav', 'rb'))



        chart = alt.Chart(source).mark_circle(size=60).encode(
            x='Horsepower',
            y='Miles_per_Gallon',
            color='Origin',
            tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
        ).interactive()

        st.altair_chart(chart)
    
    if dashboard == 'another-thing':

        st.title('Another Thing')
        st.markdown("""
        Yadda Yadda Yadda [TBC]
        """)

if __name__ == "__main__":
    main()