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

wordnet_lemmatizer = WordNetLemmatizer()


def text_input_clean(text):
    tokens = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    noStops = [w for w in tokens if w not in stopwords.words('english')]
    lemmatized = [wordnet_lemmatizer.lemmatize(w) for w in noStops]
    lemmatizedStr = ' '.join(lemmatized)
    return lemmatizedStr

def find_user_on_map(user_embedding, plot_data, user_title, user_jd):
    user_plot_data = pd.DataFrame(data=user_embedding, columns=['umap_x','umap_y'])
    user_plot_data['job_title'] = user_title
    user_plot_data['job_description'] = user_jd
    user_plot_data['source'] = 'user'

    jobs_scatter = alt.Chart(plot_data).mark_circle(size=60).encode(
        alt.X('umap_x:Q'),
        alt.Y('umap_y:Q'),
        tooltip=['job_title'],
        color='source'
    )

    user_scatter = alt.Chart(user_plot_data).mark_circle(size=60).encode(
        alt.X('umap_x:Q'),
        alt.Y('umap_y:Q'),
        tooltip=['job_title'],
        color='source',
        # labels='job_title'
    )

    scatter = alt.layer(jobs_scatter,user_scatter)

    chart = scatter.properties(
        title='Job Map!',
        width=600,
        height=400
    ).interactive()

    return chart



def main():
    st.sidebar.title('Job Atlas')
    # st.sidebar.markdown("You must unlearn what you have learned…No! Try not. Do. Or do not. There is no try.")
    dashboards = ['Where are you now?','Where do you want to go?']
    dashboard = st.sidebar.selectbox("What do you want to do?", dashboards, index=0)

    if dashboard == 'Where are you now?':

        st.title('Find yourself on the job map')
        # load data
        job_embeddings = pd.read_csv('./static/job_embeddings.csv')
        reducer = pickle.load(open('./static/reducer.sav', 'rb'))
        tfidf_vectorizer = pickle.load(open('./static/tfidf_vectorizer.sav', 'rb'))

        # Get user input
        user_title = st.text_input("Tell us your job title", value='Data Scientist')
        user_jd = st.text_input(
            "Give us a short description of what you do",
            value = 'Use data to support and automate business decisions',
            max_chars=600
            )
        user_input = user_title + ' ' + user_jd
        user_input_clean = text_input_clean(user_input)

        # vectorize and collapse
        user_vector = tfidf_vectorizer.transform([user_input_clean])
        user_embedding = reducer.transform(user_vector)

        # get altair chart
        atlas = find_user_on_map(user_embedding,job_embeddings, user_title, user_jd)
        st.altair_chart(atlas)
    
    if dashboard == 'another-thing':

        st.title('Another Thing')
        st.markdown("""
        Yadda Yadda Yadda [TBC]
        """)

if __name__ == "__main__":
    main()