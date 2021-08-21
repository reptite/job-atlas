import streamlit as st
import altair as alt
from vega_datasets import data


def main():
    st.sidebar.title('Yoda')
    st.sidebar.markdown("You must unlearn what you have learned…No! Try not. Do. Or do not. There is no try.")
    dashboards = ['chart-example','another-thing']
    dashboard = st.sidebar.selectbox("Select your Dashboard", dashboards, index=0)

    if dashboard == 'chart-example':

        st.title('Chart Example')
    
        source = data.cars()

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