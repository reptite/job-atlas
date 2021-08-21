# setup
# first load is slow - can we cache the nltk downloads?
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import math

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import pickle
import ssl

wordnet_lemmatizer = WordNetLemmatizer()

@st.cache
def nltk_downloads():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


@st.cache
def gather_data():

    print("gather_data")

    # ssl._create_default_https_context = ssl._create_unverified_context
    # ivi_data = pd.read_excel(
    #     'https://lmip.gov.au/PortalFile.axd?FieldID=2790178&.xlsx'
    #     ,sheet_name='4 digit 3 month average'
    #     )
    # cc_data = pd.read_excel(
    #        'https://www.nationalskillscommission.gov.au/sites/default/files/2021-07/Australian%20Skills%20Classification%2012-03-2021.xlsx'
    #        ,sheet_name='Core_competencies'
    #        )

    ivi_data = pd.read_csv('./static/IVI_vacancies_data.csv')
    cc_data = pd.read_csv('./static/core_competencies_by_job.csv')
    
    print("gather_data COMPLETE")
    return cc_data, ivi_data


def join_data(cc_data=None,ivi_data=None,state="AUST"):

    print("join_data {}".format(state))
    if cc_data is None: cc_data,ivi_data = gather_data()

    skills = cc_data.pivot(index="ANZSCO_Code", 
                           columns="Core_Competencies", 
                           values="Score" )
    # skills.head()

    # Add the job titles back
    u = cc_data.drop_duplicates(subset="ANZSCO_Code")[["ANZSCO_Code","ANZSCO_Title"]]
    skills = skills.join(u.set_index("ANZSCO_Code"))
    # print(skills)

    # Get also the most recent vacencies data

    u = ivi_data.loc[ivi_data['state'] == state].iloc[:, [0]+[-1]]
    u = u.rename(columns={u.columns[0]:"ANZSCO_Code", u.columns[1]:"vacancies"})
    u["ANZSCO_Code"] = pd.to_numeric(u["ANZSCO_Code"],errors="coerce")
    u["vacancies"] = pd.to_numeric(u["vacancies"],errors="coerce")
    u = u.drop_duplicates(subset="ANZSCO_Code").set_index("ANZSCO_Code")

    # print(u.index[0].__class__)
    # print(skills.index[0].__class__)
    # print(u)
    # print(skills)

    skills = skills.join(u,on="ANZSCO_Code")
    print("join_data COMPLETE")

    # print(skills.isna().sum())

    return skills



def get_retrain_distance(skills, index):

    print("get_distance {}".format(index))
    delta = lambda A,B : sum( [math.pow(2,a[1] - b[1])-1 for a,b in \
                               zip(A.iteritems(),B.iteritems()) if type(a[1]) == np.int64])

    # print("delta [1111] is {}".format(delta(skills.loc[1111],skills.loc[index])))
    # print("for A:")
    # for a in skills.iloc: print(a)
    # print(skills.loc)
    # print("for A:")

    skills['retrain'] = [ delta(a,skills.loc[index]) for a in skills.iloc ] 

    return skills


def interactive_job_details(skills=None, index=None):

    print("interactive_job_map {}".format(index))
    if index is None: index = 5412 
    if skills is None: 
        skills = join_data()

    skills = get_retrain_distance(skills,index)

    my_skills = skills.loc[[index]]
    dir(skills.columns[1:10])
    headers = skills.columns[1:10].tolist()
    # headers.to_native_types()

    skills = skills[skills["vacancies"] > 0]
    

    brush = alt.selection(type='multi', on='mouseover', nearest=True, resolve='global')

    c1 = alt.Chart(skills).mark_point(size=20).encode( 
        alt.X('retrain:Q', scale=alt.Scale(type='linear')),
        alt.Y('vacancies:Q', scale=alt.Scale(type='log')),
        color=alt.condition(brush, alt.Color('vacancies:Q', scale=alt.Scale(scheme="inferno")), alt.ColorValue('gray')),
        tooltip=["ANZSCO_Title:N"],
    ).add_selection(brush) + alt.Chart(my_skills).mark_point(size=100).encode( 
        alt.X('retrain:Q', scale=alt.Scale(type='linear')),
        alt.Y('vacancies:Q', scale=alt.Scale(type='log')),
        color=alt.ColorValue('red'),
        tooltip=['ANZSCO_Title:N'],
    ) + alt.Chart(my_skills).mark_text(align='left',dx=8).encode(
          alt.X('retrain:Q', scale=alt.Scale(type='linear')),
          alt.Y('vacancies:Q', scale=alt.Scale(type='log')),
          text='ANZSCO_Title:N',
          color=alt.ColorValue('red')
    )

    c1 = c1.interactive()

    c2 = alt.Chart(skills
    ).transform_fold( headers
    ).mark_line().encode(
        x='key:N',
        y='value:Q',
        color="vacancies:Q",
        tooltip=["ANZSCO_Title:N"],
        opacity=alt.condition(brush, alt.value(0.9), alt.value(0.02))
    ) + alt.Chart(my_skills
    ).transform_fold( headers
    ).mark_line().encode(
        x='key:N',
        y='value:Q',
        color = alt.ColorValue("red"),
        tooltip=["ANZSCO_Title:N"],
    )

    return c1 & c2.properties(width=500)


# Geoff 

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
    dashboards = ['What is your current job?','What do you want to do next?']
    dashboard = st.sidebar.selectbox("What do you want to do?", dashboards, index=0)

    cc,vaco = gather_data()   
    skills = join_data(cc,vaco,state='AUST')

    job_titles = skills.sort_values('ANZSCO_Title').ANZSCO_Title.unique().tolist()
    default_job_titles = job_titles.index('Information Officers')

    st.sidebar.text("")
    desired_job  = st.sidebar.selectbox("Select your dream job",job_titles,index=default_job_titles)
    desired_idx = skills.index[skills.ANZSCO_Title==desired_job].tolist()[0]

    print(desired_job)
    print(desired_idx)

    if dashboard == 'What is your current job?':

        st.title('Find yourself on the job map')
        # load data
        job_embeddings = pd.read_csv('./static/job_embeddings.csv')
        reducer = pickle.load(open('./static/reducer.sav', 'rb'))
        tfidf_vectorizer = pickle.load(open('./static/tfidf_vectorizer.sav', 'rb'))

        # Get user input
        nltk_downloads()
        user_title = st.text_input("Tell us your job title", value='Data Scientist')
        user_jd = st.text_input(
            "Give us a short description of what you do",
            value = 'Use data and machine learning to support and automate business decisions. Mixture of information technology, math, and business.',
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
    
    if dashboard == 'What do you want to do next?':

        state_key = ['all Australia','NSW','VIC','QLD','SA','WA','TAS','NT','ACT']
        state_key = st.selectbox("Where do you want to do it?", state_key, index=0)

        if state_key == "all Australia": state_key = "AUST"
        skills = join_data(cc,vaco,state=state_key)

        # st.title(skills['ANZSCO_Title'].iloc[index])
        # skills = get_retrain_distance(skills,index)

        st.altair_chart( interactive_job_details(skills,desired_idx) )
        

if __name__ == "__main__":
    main()