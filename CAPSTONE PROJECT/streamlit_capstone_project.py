import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import plotly.express as px

#################################################################################################
# code from https://github.com/BugzTheBunny/streamlit_logging_output_example/blob/main/app.py
#import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from contextlib import contextmanager
from io import StringIO
import sys
import logging
import time


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    "this will show the prints"
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    "This will show the logging"
    with st_redirect(sys.stderr, dst):
        yield
#################################################################################################

        
##
##data = load_iris()
##X = data.data
##y_actual = data.target
##
##df = pd.DataFrame(X)
##df.columns = data.feature_names
##df['y_actual'] = y_actual
##df = df.merge(pd.DataFrame(list(zip([0,1,2],data.target_names)),columns =['y_actual','Target_actual']), on='y_actual', how='left')
##
##from sklearn.cluster import KMeans
##kmeans_cluster = KMeans(n_clusters=3, random_state=0).fit(X)
##y_predicted = kmeans_cluster.labels_
##
##df['y_predicted'] = y_predicted
##
##df = df.merge(pd.DataFrame(list(zip([1,0,2],data.target_names)),columns =['y_predicted','Target_predicted']), on='y_predicted', how='left')
##df['Target_actual, Target_predicted']= df['Target_actual']+', '+df['Target_predicted']
##
##cc_df = pd.DataFrame(kmeans_cluster.cluster_centers_)
##cc_df.columns = data.feature_names
##cc_df['y_actual'] = [0,1,2]
##cc_df = cc_df.merge(pd.DataFrame(list(zip([0,1,2],data.target_names)),columns =['y_actual','Target_actual']), on='y_actual', how='left')
##cc_df['y_predicted'] = [0,1,2]
##cc_df = cc_df.merge(pd.DataFrame(list(zip([0,1,2],data.target_names)),columns =['y_predicted','Target_predicted']), on='y_predicted', how='left')
##cc_df['Target_actual, Target_predicted']= cc_df['Target_actual']+' mean'
##
##df_extended = pd.concat([df, cc_df], ignore_index=True)
##
##
##
##target_list = data.feature_names


st.title('DSA Capstone Project')
st.subheader('Titus Teodorescu/ALTRD')

st.markdown('Task: Please use your imagination to show your insights from the data in `data_capstone_dsa2021_2022.csv` and tell a cogent story to your audience.')

st.write('''Here\'s the information that was provided about the data.''')
st.info('''
* The data is a table that records response time, scores to questions, and some demographic info.
* Each row is a participant.
* Column meaning:
    * `rt_gs_i`: the response time to *i*th item in seconds. Note that here is no response time for the first item.
    * `rt_total`: total reponse time by the participant
    * `gs_i`: score of the *i*th item.
    * `sum_score`: total score of the participant
    * `gender`
    * `home_computer`: whether the participant has computer at home
    * `state`: the state the participant is living
    * `age`''')

st.subheader('First things first')
st.write('Let\'s import some packages, upload the csv file to a panda dataframe, and take a look at the data.')

st.code('''
import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.pandas.read_csv('data_capstone_dsa2021_2022.csv')
''')

##import streamlit as st
##from sklearn.datasets import load_iris
##import pandas as pd
##import numpy as np
##import plotly.express as px

df = pd.pandas.read_csv('data_capstone_dsa2021_2022.csv')

st.write('Here are the first 5 rows in the data frame.')

st.table(data=df[0:5])

st.write('Let\'s expore the data.')
st.code('''
print(df.info())
''')

st.code('''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1169 entries, 0 to 1168
Data columns (total 45 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   rt_gs_2        1169 non-null   int64 
 1   rt_gs_3        1169 non-null   int64 
 2   rt_gs_4        1169 non-null   int64 
 3   rt_gs_5        1169 non-null   int64 
 4   rt_gs_6        1169 non-null   int64 
 5   rt_gs_7        1169 non-null   int64 
 6   rt_gs_8        1169 non-null   int64 
 7   rt_gs_9        1169 non-null   int64 
 8   rt_gs_10       1169 non-null   int64 
 9   rt_gs_11       1169 non-null   int64 
 10  rt_gs_12       1169 non-null   int64 
 11  rt_gs_13       1169 non-null   int64 
 12  rt_gs_14       1169 non-null   int64 
 13  rt_gs_15       1169 non-null   int64 
 14  rt_gs_16       1169 non-null   int64 
 15  rt_gs_17       1169 non-null   int64 
 16  rt_gs_18       1169 non-null   int64 
 17  rt_gs_19       1169 non-null   int64 
 18  rt_gs_20       1169 non-null   int64 
 19  rt_total       1169 non-null   int64 
 20  gs_1           1169 non-null   int64 
 21  gs_2           1169 non-null   int64 
 22  gs_3           1169 non-null   int64 
 23  gs_4           1169 non-null   int64 
 24  gs_5           1169 non-null   int64 
 25  gs_6           1169 non-null   int64 
 26  gs_7           1169 non-null   int64 
 27  gs_8           1169 non-null   int64 
 28  gs_9           1169 non-null   int64 
 29  gs_10          1169 non-null   int64 
 30  gs_11          1169 non-null   int64 
 31  gs_12          1169 non-null   int64 
 32  gs_13          1169 non-null   int64 
 33  gs_14          1169 non-null   int64 
 34  gs_15          1169 non-null   int64 
 35  gs_16          1169 non-null   int64 
 36  gs_17          1169 non-null   int64 
 37  gs_18          1169 non-null   int64 
 38  gs_19          1169 non-null   int64 
 39  gs_20          1169 non-null   int64 
 40  sum_score      1169 non-null   int64 
 41  gender         1169 non-null   object
 42  home_computer  1169 non-null   object
 43  state          1169 non-null   object
 44  age            1169 non-null   int64 
dtypes: int64(42), object(3)
memory usage: 411.1+ KB
''')


st.write('''
Let\'s try to answer some basic questions:
* How many rows (participants) are there?
    * There are 1169 rows (participants).
* How many columns are there?
    * There are 45 columns.
* Are there any missing entries (i.e., `None` or `NaN` values)?
    * No, for each row, the number of non-null objects is 1169, which matches the number of rows, 1169.
* What are the data types of the columns?
    * Except for three columns (`gender`, `home_computer`, and `state`), which are `object` data type, all other columns are `int64` data type.  
* How many items are there?
    * Based on the previous output, which contains 20 entries of the type `gs_i`, there are 20 items.
''')





   
st.write('''
* Is `rt_gs_1` missing because all its values are 0 ? Or is it missing for other reasons and we need to compute the value of `rt_gs_1` from all the other columns?
    * Based on the code below, `rt_gs_1` is missing because its value is 0 for all the rows.
''')
st.code('''
df['rt_gs_1'] = df['rt_total']-df['rt_gs_2']-df['rt_gs_3']-df['rt_gs_4']-df['rt_gs_5']-df['rt_gs_6']-df['rt_gs_7']-df['rt_gs_8']-df['rt_gs_9']-df['rt_gs_10']-df['rt_gs_11']-df['rt_gs_12']-df['rt_gs_13']-df['rt_gs_14']-df['rt_gs_15']-df['rt_gs_16']-df['rt_gs_17']-df['rt_gs_18']-df['rt_gs_19']-df['rt_gs_20']
print(df[df['rt_gs_1']!=0].shape)
''')

st.code('''
(0, 46)
''')

st.write('''
* How does it mean that value of `rt_gs_1` is 0 for all rows?
    * It means that all participants took 0 seconds to respond to item #1.
* Does that make any sense?
    * I\'ve never see an item that can be answered in 0 seconds. Note that, based on the code below, only 983 of the 1169 participants answered item #1 correctly.
''')
st.code('''
from collections import Counter
print(Counter(df['gs_1'].T.to_numpy('int').tolist()))
''')
st.code('''
Counter({1: 983, 0: 186})
''')

st.write('''
* Is `sum_score` indeed the sum of all the values of `gs_i` ?
    * Yes, based on the code below, `sum_score` is indeed the sum of all the values of `gs_i`.
''')
st.code('''
df['correct_sum'] = df['gs_1']+df['gs_2']+df['gs_3']+df['gs_4']+df['gs_5']+df['gs_6']+df['gs_7']+df['gs_8']+df['gs_9']+df['gs_10']+df['gs_11']+df['gs_12']+df['gs_13']+df['gs_14']+df['gs_15']+df['gs_16']+df['gs_17']+df['gs_18']+df['gs_19']+df['gs_20']-df['sum_score']
print(df[df['correct_sum']!=0].shape)
''')
st.code('''
(0, 47)
''')




st.write('')
st.write('')
st.write('')
st.write('')

##st.sidebar.subheader("Input for the interactive 3d scatterplot")
##select_x_3d = st.sidebar.selectbox("Select x feature: ", target_list, 0)
##select_y_3d = st.sidebar.selectbox("Select y feature: ", target_list, 1)
##select_z_3d = st.sidebar.selectbox("Select z feature: ", target_list, 2)
##
##
##fig_3d = px.scatter_3d(df_extended, x=select_x_3d, y=select_y_3d, z=select_z_3d, color='Target_actual, Target_predicted', width=900, height=600)
###fig_3d.show()
##
##st.subheader('Interactive 3D scatterplot')
##st.markdown('Select the x, y, and z features from the appropriate dropdowns in the side bar. The selections will be reflected in the display below.')
##st.text('(x,y,z)=(\''+select_x_3d + '\',\''+select_y_3d+'\',\''+select_z_3d+'\')')
##st.markdown('Both raw data and the mean of the clusters are shown below.')
##
##st.plotly_chart(fig_3d, use_container_width=True)
##
##
##st.sidebar.subheader("Input for the interactive 2d scatterplots")
##select_x_2d = st.sidebar.selectbox("Select x feature : ", target_list, 0)
##select_y_2d = st.sidebar.selectbox("Select y feature : ", target_list, 1)
##
##
##
##fig_2d = px.scatter(df_extended, x=select_x_2d, y=select_y_2d, color='Target_actual, Target_predicted', width=900, height=600)
###fig_3d.show()
##
##st.subheader('First interactive 2D scatterplot')
##st.markdown('Select the x and y features from the appropriate dropdowns in the side bar. The selections will be reflected in the display below.')
##st.text('(x,y)=(\''+select_x_2d + '\',\''+select_y_2d+'\')')
##st.markdown('Both raw data and the mean of the clusters are shown below.')
##
##st.plotly_chart(fig_2d, use_container_width=True)
##
##fig_2d_second = px.scatter(df, x=select_x_2d, y=select_y_2d, color='Target_actual, Target_predicted', facet_row='Target_actual', facet_col='Target_predicted', width=900, height=900)
###fig_2d.show()
###fig_3d.show()
##
##st.subheader('Second interactive 2D scatterplot')
##st.markdown('Select the x and y features from the appropriate dropdowns in the side bar. The selections will be reflected in the display below.')
##st.text('(x,y)=(\''+select_x_2d + '\',\''+select_y_2d+'\')')
##st.text('facet_row=\'Target_actual\'')
##st.text('facet_col=\'Target_predicted\'')
##
##st.plotly_chart(fig_2d_second, use_container_width=True)
##




##def demo_function():
##    """
##    Just a sample function to show how it works.
##    :return:
##    """
##    for i in range(10):
##        logging.warning(f'Counting... {i}')
##        time.sleep(2)
##        print('Time out...')
##
##
##if __name__ == '__main__':
##    with st_stdout("success"), st_stderr("code"):
##        demo_function()











