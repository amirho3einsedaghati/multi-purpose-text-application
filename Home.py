import streamlit as st
from configs.db_configs import create_table, engine
import plotly.express as px
import pandas as pd
from streamlit.components.v1 import html
from configs.html_features import set_image



def main():
    create_table()
    st.title('Welcome to the Multi-purpose Text Application')
    im1, im2, im3 = st.columns([1, 5.3, 1])
    with im1:
        pass
    with im2:
        url = "https://i.postimg.cc/jdF1hPng/combined.png"
        html(set_image(url), height=400, width=400)
    with im3:
        pass

    col11, col12, col13, col14, col15, col16 = st.columns([1, 1, 2.2, 2.2, 1.3, 2])
    with col11:
        pass

    with col12:
        pass

    with col13:
        st.info('Text Summarizer')

    with col14:
        st.info('Text Analyzer')

    with col15:
        pass
    
    with col16:
        pass

    col21, col22, col23, col24, col25, col26 = st.columns([1, 1, 2.2, 2.2, 1.3, 2])
    with col21:
        pass

    with col22:
        pass

    with col23:
        st.info('Text Translator')
        
    with col24:
        st.info('Topic Modeling')

    with col25:
        pass
    
    with col26:
        pass
    
    plot1, plot2, plot3 = st.columns([5,1,5])
    df = pd.read_sql('SELECT app from input_text', engine)['app'].value_counts().to_frame().reset_index()
    df.columns = ['App', 'Frequency']
    custom_color = px.colors.sequential.Blues_r

    with plot1:
        fig1 = px.pie(df, 'App', 'Frequency', title='The frequency of service usage as a percentage', width=400, height=500, color_discrete_sequence=custom_color)
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1)

    with plot2:
        pass
    
    with plot3:
        fig2 = px.area(df, 'App', 'Frequency', width=400, height=500, title='The frequency of service usage')
        st.plotly_chart(fig2)


if __name__ == '__main__':
    main()