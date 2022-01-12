import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import plotly.express as px

data = load_iris()
X = data.data
y_actual = data.target

df = pd.DataFrame(X)
df.columns = data.feature_names
df['y_actual'] = y_actual
df = df.merge(pd.DataFrame(list(zip([0,1,2],data.target_names)),columns =['y_actual','Target_actual']), on='y_actual', how='left')

from sklearn.cluster import KMeans
kmeans_cluster = KMeans(n_clusters=3, random_state=0).fit(X)
y_predicted = kmeans_cluster.labels_

df['y_predicted'] = y_predicted

df = df.merge(pd.DataFrame(list(zip([1,0,2],data.target_names)),columns =['y_predicted','Target_predicted']), on='y_predicted', how='left')
df['Target_actual, Target_predicted']= df['Target_actual']+', '+df['Target_predicted']

cc_df = pd.DataFrame(kmeans_cluster.cluster_centers_)
cc_df.columns = data.feature_names
cc_df['y_actual'] = [0,1,2]
cc_df = cc_df.merge(pd.DataFrame(list(zip([0,1,2],data.target_names)),columns =['y_actual','Target_actual']), on='y_actual', how='left')
cc_df['y_predicted'] = [0,1,2]
cc_df = cc_df.merge(pd.DataFrame(list(zip([0,1,2],data.target_names)),columns =['y_predicted','Target_predicted']), on='y_predicted', how='left')
cc_df['Target_actual, Target_predicted']= cc_df['Target_actual']+' mean'

df_extended = pd.concat([df, cc_df], ignore_index=True)



target_list = data.feature_names


st.title('DSA Homework 6 / Titus Teodorescu')
st.markdown('This streamlit dashboard presents several ways to visualize the clusters and to interact with the data in the IRIS data set. The dashboard contains one 3D scatterplot and two 2D scatterplots.')

st.sidebar.subheader("Input for the interactive 3d scatterplot")
select_x_3d = st.sidebar.selectbox("Select x feature: ", target_list, 0)
select_y_3d = st.sidebar.selectbox("Select y feature: ", target_list, 1)
select_z_3d = st.sidebar.selectbox("Select z feature: ", target_list, 2)


fig_3d = px.scatter_3d(df_extended, x=select_x_3d, y=select_y_3d, z=select_z_3d, color='Target_actual, Target_predicted', width=900, height=600)
#fig_3d.show()

st.subheader('Interactive 3D scatterplot')
st.markdown('Select the x, y, and z features from the appropriate dropdowns in the side bar. The selections will be reflected in the display below.')
st.text('(x,y,z)=(\''+select_x_3d + '\',\''+select_y_3d+'\',\''+select_z_3d+'\')')
st.markdown('Both raw data and the mean of the clusters are shown below.')

st.plotly_chart(fig_3d, use_container_width=True)


st.sidebar.subheader("Input for the interactive 2d scatterplots")
select_x_2d = st.sidebar.selectbox("Select x feature : ", target_list, 0)
select_y_2d = st.sidebar.selectbox("Select y feature : ", target_list, 1)



fig_2d = px.scatter(df_extended, x=select_x_2d, y=select_y_2d, color='Target_actual, Target_predicted', width=900, height=600)
#fig_3d.show()

st.subheader('First interactive 2D scatterplot')
st.markdown('Select the x and y features from the appropriate dropdowns in the side bar. The selections will be reflected in the display below.')
st.text('(x,y)=(\''+select_x_2d + '\',\''+select_y_2d+'\')')
st.markdown('Both raw data and the mean of the clusters are shown below.')

st.plotly_chart(fig_2d, use_container_width=True)

fig_2d_second = px.scatter(df, x=select_x_2d, y=select_y_2d, color='Target_actual, Target_predicted', facet_row='Target_actual', facet_col='Target_predicted', width=900, height=900)
#fig_2d.show()
#fig_3d.show()

st.subheader('Second interactive 2D scatterplot')
st.markdown('Select the x and y features from the appropriate dropdowns in the side bar. The selections will be reflected in the display below.')
st.text('(x,y)=(\''+select_x_2d + '\',\''+select_y_2d+'\')')
st.text('facet_row=\'Target_actual\'')
st.text('facet_col=\'Target_predicted\'')

st.plotly_chart(fig_2d_second, use_container_width=True)

