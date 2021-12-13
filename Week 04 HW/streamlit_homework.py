import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


df = pd.read_csv('data for week4/science_response.csv')
df[['startTime']] = df[['startTime']].apply(pd.to_datetime)
df[['playerID', 'gender']] = df[['playerID', 'gender']].astype("string")

item_list = ['item' + str(sub) for sub in list(range(1, 41))]
gender_list = df['gender'].unique()

#st.title('Streamlit Dashboard')

select_gender = st.sidebar.selectbox("Select gender: ", gender_list)
select_item = st.sidebar.selectbox("Select item: ", item_list)


fig = px.histogram(df.query('gender==@select_gender'), x='sum_score', color=select_item, title='Freq Distribution of Sum Scores for ' + select_gender + ' Respondants, Colored by Responses to '+ select_item)

st.plotly_chart(fig)



