import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import math

@st.cache
def gather_data():

    print("gather_data")
    ivi_data = pd.read_excel(
        'https://lmip.gov.au/PortalFile.axd?FieldID=2790178&.xlsx'
        ,sheet_name='4 digit 3 month average'
        )

    cc_data = pd.read_excel(
            'https://www.nationalskillscommission.gov.au/sites/default/files/2021-07/Australian%20Skills%20Classification%2012-03-2021.xlsx'
            ,sheet_name='Core_competencies'
            )

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

    # Get also the most recent vacencies data

    u = ivi_data.loc[ivi_data['state'] == state].iloc[:, [0]+[-1]]
    u = u.rename(columns={u.columns[0]:"ANZSCO_Code", u.columns[1]:"vacancies"})
    u = u.drop_duplicates(subset="ANZSCO_Code")

    skills = skills.join(u.set_index("ANZSCO_Code"))
    
    print("join_data COMPLETE")
    return skills



def get_retrain_distance(skills, index):

    print("get_distance {}".format(index))
    delta = lambda A,B : sum( [math.pow(2,a[1] - b[1])-1 for a,b in \
                               zip(A.iteritems(),B.iteritems()) if type(a[1]) == np.int64])

    print("delta [0] is {}".format(delta(skills.iloc[0],skills.iloc[index])))

    skills['retrain'] = [ delta(a,skills.iloc[index]) for a in skills.iloc ] 

    return skills


def interactive_job_details(skills=None, index=None):

    print("interactive_job_map {}".format(index))
    if index is None: index = 100
    if skills is None: 
        skills = join_data()

    skills = get_retrain_distance(skills,index)

    my_skills = skills.iloc[[index]]
    dir(skills.columns[1:10])
    headers = skills.columns[1:10].tolist()
    # headers.to_native_types()


    brush = alt.selection(type='multi', on='mouseover', nearest=True, resolve='global')

    c1 = alt.Chart(skills).mark_point().encode( 
        alt.X('retrain:Q', scale=alt.Scale(type='linear')),
        alt.Y('vacancies:Q', scale=alt.Scale(type='log')),
        color=alt.condition(brush, alt.Color('vacancies:Q', scale=alt.Scale(scheme="inferno")), alt.ColorValue('gray')),
        tooltip=["ANZSCO_Title:N"],
    ).add_selection(brush) + alt.Chart(my_skills).mark_point().encode( 
        alt.X('retrain:Q', scale=alt.Scale(type='linear')),
        alt.Y('vacancies:Q', scale=alt.Scale(type='log')),
        color=alt.ColorValue("red"),
        tooltip=["ANZSCO_Title:N"],
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



def main():
    st.sidebar.title('Yoda')
    st.sidebar.markdown("You must unlearn what you have learned…No! Try not. Do. Or do not. There is no try.")
    dashboards = ['chart-example','another-thing']
    dashboard = st.sidebar.selectbox("Select your Dashboard", dashboards, index=0)

    cc,vaco = gather_data()
    skills = join_data(cc,vaco,state='AUST')

    if dashboard == 'chart-example':

        st.title('Chart Example')
    
        # source = data.cars()

        chart = alt.Chart(skills).mark_circle(size=60).encode(
            x='Horsepower',
            y='Miles_per_Gallon',
            color='Origin',
            tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
        ).interactive()

        st.altair_chart(chart)
    
    if dashboard == 'another-thing':

        index = 142
        st.title(skills['ANZSCO_Title'].iloc[index])
        # skills = get_retrain_distance(skills,index)

        st.altair_chart(interactive_job_details(skills,index))
        
        st.markdown("""
        Yadda Yadda Yadda [TBC]
        """)

if __name__ == "__main__":
    main()