import streamlit as st
import pandas as pd
import numpy as np
import re
#from fuzzywuzzy import fuzz, process
#import plotly.express as px

#################################################################################################
# code from https://github.com/BugzTheBunny/streamlit_logging_output_example/blob/main/app.py

# from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
# from threading import current_thread
# from contextlib import contextmanager
# from io import StringIO
# import sys
# import logging
# import time
#
#
# @contextmanager
# def st_redirect(src, dst):
#     placeholder = st.empty()
#     output_func = getattr(placeholder, dst)
#
#     with StringIO() as buffer:
#         old_write = src.write
#
#         def new_write(b):
#             if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
#                 buffer.write(b + '')
#                 output_func(buffer.getvalue() + '')
#             else:
#                 old_write(b)
#
#         try:
#             src.write = new_write
#             yield
#         finally:
#             src.write = old_write
#
#
# @contextmanager
# def st_stdout(dst):
#     "this will show the prints"
#     with st_redirect(sys.stdout, dst):
#         yield
#
#
# @contextmanager
# def st_stderr(dst):
#     "This will show the logging"
#     with st_redirect(sys.stderr, dst):
#         yield
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

st.write('The project is broken into three parts. In part 1, I\'ll take a quick look at the data; in part 2, I\'ll do some data clean up; and in part 3, I\'ll do some exploratory data analysis and create a few pretty graphs.')
st.subheader('Part 1. Quick data overview')
st.write('Let\'s import some packages, upload the csv file to a panda dataframe, and take a quick look at the data.')

st.code('''
import streamlit as st
import pandas as pd
import numpy as np
import re
#from fuzzywuzzy import fuzz, process
#import plotly.express as px

df = pd.read_csv('CAPSTONE-PROJECT/data_capstone_dsa2021_2022.csv')
''')

##import streamlit as st
##import pandas as pd
##import numpy as np
##import plotly.express as px

df = pd.read_csv('CAPSTONE-PROJECT/data_capstone_dsa2021_2022.csv')   # use this for stream cloud
#df = pd.read_csv('data_capstone_dsa2021_2022.csv')                     # use this locally

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


st.subheader('Part 2. Clean up the data')
st.write('What makes data dirty? Many things: missing data, inconsistent data, duplicate data, etc.')

st.write('''
* Are there missing data?
    * No, as mentioned before. The code below shows another way to conclude that no data are missing.
''')
st.code('''
df.isnull().sum()
''')
st.code('''
rt_gs_2          0
rt_gs_3          0
rt_gs_4          0
rt_gs_5          0
rt_gs_6          0
rt_gs_7          0
rt_gs_8          0
rt_gs_9          0
rt_gs_10         0
rt_gs_11         0
rt_gs_12         0
rt_gs_13         0
rt_gs_14         0
rt_gs_15         0
rt_gs_16         0
rt_gs_17         0
rt_gs_18         0
rt_gs_19         0
rt_gs_20         0
rt_total         0
gs_1             0
gs_2             0
gs_3             0
gs_4             0
gs_5             0
gs_6             0
gs_7             0
gs_8             0
gs_9             0
gs_10            0
gs_11            0
gs_12            0
gs_13            0
gs_14            0
gs_15            0
gs_16            0
gs_17            0
gs_18            0
gs_19            0
gs_20            0
sum_score        0
gender           0
home_computer    0
state            0
age              0
rt_gs_1          0
correct_sum      0
dtype: int64
''')

st.write('''
* The two columns, `rt_gs_1 ` and `correct_sum`, introduced in part 1 are no longer needed. Can you delete them?
    * Yes, the code block below shows one way to do it.
''')
st.code('''
columns_to_delete = ['rt_gs_1','correct_sum']
df.drop(columns_to_delete, inplace=True, axis=1)
df.columns
''')
st.code('''
Index(['rt_gs_2', 'rt_gs_3', 'rt_gs_4', 'rt_gs_5', 'rt_gs_6', 'rt_gs_7',
       'rt_gs_8', 'rt_gs_9', 'rt_gs_10', 'rt_gs_11', 'rt_gs_12', 'rt_gs_13',
       'rt_gs_14', 'rt_gs_15', 'rt_gs_16', 'rt_gs_17', 'rt_gs_18', 'rt_gs_19',
       'rt_gs_20', 'rt_total', 'gs_1', 'gs_2', 'gs_3', 'gs_4', 'gs_5', 'gs_6',
       'gs_7', 'gs_8', 'gs_9', 'gs_10', 'gs_11', 'gs_12', 'gs_13', 'gs_14',
       'gs_15', 'gs_16', 'gs_17', 'gs_18', 'gs_19', 'gs_20', 'sum_score',
       'gender', 'home_computer', 'state', 'age'],
      dtype='object')
''')

st.write('''
* Can I get a descriptive summary of each column?
    * I don\'t see why not. The pandas `describe()` function gives a lot of info about the columns.
''')
st.code('''
df.describe()
''')
df_description = df.describe()
st.table(data=df_description)

st.write('''
* What can you infer from the output of `describe()` above?
    * Quick a lot, if you look carefully.
    * For example, the minimum values for the columns `rt_gs_2` through `rt_gs_20` are 0, 1, 2, and 3. If you remember that these values are durations in seconds, you can
    infer that some participants answered some questions very quickly, possibly without reading the questions, indicating lack of interest from some participants to answer
    these 20 questions.
    * When you continue to look at the columns `rt_gs_2` through `rt_gs_20` but this time you look globally at the values in the rows for min, 25%, 50%, and 75%, these values
    vary from a minimum of 0 seconds to a maximum of 69 seconds. That shows that a majority of the participants answered each questions in less than 69 seconds. Without more
    information, it\'s hard to interpret this without more information about the 20 questions. If the questions are easy, these durations could be considered reasonable. If the
    questions are hard, these durations could indicate a lack of engagement with the questions.
    * When you look at values of max in the same columns, the values vary from 248 seconds (4.13 minutes) to 3134 seconds (52.23 minutes). That could indicate that some participants
    struggled with some questions. Or it could indicate that some participants were not actively engaged with the questions (for example, they could have started a question, got
    distracted by something, and return to answer the questions at a later time. Or it could indicate that some participants had internet connectivity issues.
    * Continuing to look at the  values in the max row and in the columns `rt_gs_2` through `rt_gs_20` and comparing them with the corresponding values in the rows for mean and std,
    the numbers in the max row are at least 3 standard deviations above the mean, which indicates the presence of some wild outliers.
    * When looking at the values in the `rt_total`, the value in the max row is at least 3 standard deviations above the mean, which indicates the presence of at least one participant with
    outlier response durations.
    * Moving to the right and looking at the columns `gs_1` through `gs_20`, the min value and max value for each of the columns are 0 and 1, respectively. Recalling that the data
    type for these columns is `int64`, that\'s a confirmation that the values of `gs_1` through `gs_20` are either 0 or 1 for all participants, which is a chec that the data in these
    columns is consistent.
    * Continuing to look at these columns, since the values are either 0 or 1, it follows that the values in the mean row can be interpreted as the percent of partcipants that answered
    the questions correctly. The 3 smallest mean values are 0.5603, 0.7083, and 0.7263, with all other mean values greater than 0.8. Thse values shows that a large majority of the
    participants answered the questions correctly.
    * The same conclusion can be drawn when you look at the values in the `sum_score` column, where the min value is at least 3 standarad deviations lower than the mean.
    * Looking at the values in the `age` column, the min value is at least 3 standard deviations below the mean, while the max value is at least 3 standard deviations above the mean.
    * In the same column, `age`, the min value is 0. Such a value is not reasonable for describing the age of a participant, which could indicate that the age of some participants were not
    available and were replaced by a default value of 0. We might need to clean the data in the `age` column. One way to clean the data is to replace the 0 value with some other values, such
    as the mean age.
''')

st.write('''
* Can I get the frequency counts for the columns `gs_1` through `gs_20` ? I want to double check that the these columns only contains 0s and 1s.
    * Sure. The output below shows that the constraints on the data in these columns are respected.
''')
st.code('''
for column in ['gs_1', 'gs_2', 'gs_3', 'gs_4', 'gs_5', 'gs_6', 'gs_7', 'gs_8', 'gs_9', 'gs_10', 'gs_11', 'gs_12', 'gs_13', 'gs_14', 'gs_15', 'gs_16', 'gs_17', 'gs_18', 'gs_19', 'gs_20']:
    print(f'{column}: {Counter(df[column].T)}')
''')
st.code('''
gs_1: Counter({1: 983, 0: 186})
gs_2: Counter({1: 1025, 0: 144})
gs_3: Counter({1: 828, 0: 341})
gs_4: Counter({1: 1025, 0: 144})
gs_5: Counter({1: 1118, 0: 51})
gs_6: Counter({1: 1025, 0: 144})
gs_7: Counter({1: 948, 0: 221})
gs_8: Counter({1: 1011, 0: 158})
gs_9: Counter({1: 938, 0: 231})
gs_10: Counter({1: 1012, 0: 157})
gs_11: Counter({1: 942, 0: 227})
gs_12: Counter({1: 1036, 0: 133})
gs_13: Counter({1: 655, 0: 514})
gs_14: Counter({1: 1068, 0: 101})
gs_15: Counter({1: 1165, 0: 4})
gs_16: Counter({1: 1090, 0: 79})
gs_17: Counter({1: 849, 0: 320})
gs_18: Counter({1: 1067, 0: 102})
gs_19: Counter({1: 1073, 0: 96})
gs_20: Counter({1: 1160, 0: 9})
''')

st.write('''
* The output of `describe()` did not include any information about the non-integer columns: `gender`, `home_computer`, and `state`. What can you say about these three columns?
    * Let\'s look at the frequency couns for these columns, as shown in the output of the code below.
    * Slightly more than half of the participants identifed as females.
    * More than two thirds of the participants have a computer at home.
    * The `state` column is in a bad shape and the values in this column need to be normalized.
''')
st.code('''
for column in ['gender', 'home_computer', 'state']:
    print(f'{column}: {Counter(df[column].T)}')
''')
st.code('''
gender: Counter({'Female': 598, 'Male': 571})
home_computer: Counter({'Yes': 792, 'No': 377})
state: Counter({'California': 81, 'Florida': 51, 'Texas': 51, 'New York': 42, 'Ohio': 37, 'Michigan': 34, 'Illinois': 27, 'Pennsylvania': 26, 'Indiana': 22, 'North Carolina': 20, 'Georgia': 18, 'Wisconsin': 18, 'Maryland': 16, 'Missouri': 16, 'Kentucky': 15, 'New Jersey': 15, 'USA': 15, 'Louisiana': 15, 'Virginia': 13, 'PA': 13, 'United States': 12, 'Tennessee': 12, 'Massachusetts': 11, 'Arizona': 9, 'Alabama': 9, 'Colorado': 9, 'florida': 8, 'Minnesota': 8, 'Washington': 8, 'NY': 8, 'MN': 7, 'Oregon': 7, 'CA': 7, 'Connecticut': 7, 'california': 7, 'West Virginia': 6, 'ohio': 6, 'NJ': 6, 'NC': 6, 'Oklahoma': 6, 'Kansas': 6, 'michigan': 6, 'Washington State': 5, 'Canada': 5, 'texas': 5, 'South Carolina': 5, 'Missouri, USA': 5, 'Iowa': 5, 'Mississippi': 4, 'maryland': 4, 'NH': 4, 'US': 4, 'washington': 4, 'Nebraska': 4, 'NY, USA': 4, 'TX': 4, 'WA': 4, 'connecticut': 4, 'Ohio, USA': 4, 'new jersey': 4, 'Utah': 3, 'Texas, USA': 3, 'PA, USA': 3, 'alabama': 3, 'WI': 3, 'TN': 3, 'usa': 3, 'New Hampshire': 3, 'California, USA': 3, 'WV': 3, 'Idaho': 3, 'pennsylvania': 3, 'New Mexico': 3, 'Nevada': 3, 'FL': 3, 'Michigan, USA': 3, 'North Carolina, USA': 3, 'IL': 3, 'GA': 2, 'Virginia, USA': 2, 'colorado': 2, 'tennessee': 2, 'AL': 2, 'Florida, USA': 2, 'Illinois, USA': 2, 'north carolina': 2, 'NEW YORK': 2, 'indiana': 2, 'Maryland, USA': 2, 'Oregon, USA': 2, 'Pennsylvania, USA': 2, 'SC': 2, 'Arkansas': 2, 'new york': 2, 'nc': 2, 'KY': 2, 'Montana': 2, 'NJ, USA': 2, 'minnesota': 2, 'tx': 2, 'Maine': 2, 'CO': 2, 'WISCONSIN': 2, 'AR, USA': 2, 'CT, USA': 2, 'FLORIDA': 2, 'LA': 2, 'Michigan USA': 2, 'MI': 2, 'kentucky': 2, 'VA': 2, 'nj': 2, 'Washington State, USA': 2, 'CA, USA': 2, 'wisconsin': 1, 'MO USA': 1, 'Utah, United States of America': 1, 'denver, colorado': 1, 'Indiana, USA': 1, 'idaho, usa': 1, 'FLORIDA, USA': 1, 'KS, USA': 1, 'Rogers, Arkansas': 1, 'va, usa': 1, 'Decatur, AL': 1, 'Georgia, USA': 1, 'tennesee': 1, 'canada': 1, 'Lousiana': 1, 'Vermont, United States': 1, 'Singapore': 1, 'ma usa': 1, 'Cheshire High School, Cheshire, Connecticut': 1, 'MO': 1, 'utah': 1, 'Lakewood, CO': 1, 'Hawaii': 1, 'Pennsylvania, United States': 1, 'Rhode Island': 1, 'Lakeland FL': 1, 'SC,  US': 1, 'Tampa, FL': 1, 'Santa Clara': 1, 'boston massachusetts': 1, 'Cincinnati, Ohio': 1, 'los angeles, ca': 1, 'WA State': 1, 'Atlanta, GA': 1, 'NC, USA': 1, 'Buenos Aires, Argentina': 1, 'US, NY': 1, 'sc': 1, 'Delaware': 1, 'Iowa, United States': 1, 'NEW CASTLE': 1, 'louisville ky': 1, 'SOUTH CAROLINA': 1, 'Henderson, NV': 1, 'NORTH CAROLINA': 1, 'Philippines': 1, 'Carson City': 1, 'Massachusettes': 1, 'OKlahoma': 1, 'India': 1, 'Henrico, VA, USA': 1, 'Orange County, California': 1, 'ny': 1, 'KY, USA': 1, 'VA, USA': 1, 'Utah, USA': 1, 'Paradise, California': 1, 'Pennsyvania': 1, 'Bristol, Connecticut': 1, 'illinois': 1, 'Georgia, US': 1, 'missouri': 1, 'Caflifornia': 1, 'Tennessee, USA': 1, 'AZ': 1, 'grosse pointe, michigan, united states': 1, 'usa Fla': 1, 'Dalton High, Dalton, Ga.': 1, 'port huron, michigan': 1, 'Franklin, NC': 1, 'Pearl City, HI': 1, 'Des Moines, Iowa': 1, 'St. Thomas, USVI': 1, 'lake butler florida': 1, 'hawaii': 1, 'Norfolk, VA': 1, 'Duval': 1, 'Malaysia': 1, 'md': 1, 'SD': 1, 'Germany': 1, 'richmond, IN': 1, 'Michigan (US)': 1, 'maine': 1, 'North Carolina, United States': 1, 'Arkansas, USA': 1, 'nebraska': 1, 'Cincinnati Ohio': 1, 'Alaska': 1, 'New York, USA': 1, 'texas,usa': 1, 'MA': 1, 'Livingston, Tx': 1, 'Hampshire High, Hampshire, Illinois': 1, 'United States, MN': 1, 'tennessee, usa': 1, 'California, United States': 1, 'New York Ciry': 1, 'Columbus, Ohio': 1, 'Ocala, FL': 1, 'Montgomery County, PA': 1, 'millinocket': 1, 'Framingham': 1, 'New York City': 1, 'North Florida': 1, 'youngstown oh': 1, 'Walla Walla, WA': 1, 'OH': 1, 'Flushing, Michigan': 1, 'Iowa  United States': 1, 'Slovakia': 1, 'Aruba': 1, 'Worthington': 1, 'RI': 1, 'Converse': 1, 'New York USA': 1, 'MICHIGAN': 1, 'Westfield,NJ': 1, 'Independence, MO USA': 1, 'multiple locations': 1, 'Chicago': 1, 'WI, USA': 1, 'Quartz Hill': 1, 'Whitwell, TN': 1, 'sheboygan, wi': 1, 'America': 1, 'idaho': 1, 'maryland, usa': 1, 'Michigan, US': 1, 'modern college,India': 1, 'Cathlamet, WA': 1, 'Plant City, Florida': 1, 'United States, Illinois': 1, 'Columbia, SC': 1, 'New  Jersey USA': 1, 'New York, US': 1, 'Maine, USA': 1, 'united states': 1, 'New York State': 1, 'Louisiana, USA': 1, 'CT': 1, 'Alabama, USA': 1, 'ga': 1, 'ILLINOIS': 1, 'arizona': 1, 'nyc, ny': 1, 'michigan, usa': 1, 'el paso, tx': 1, 'californi': 1, 'kansas': 1, 'Miami': 1, 'ga. usa': 1, 'Pittsburgh, Pennsylvania': 1, 'W.V': 1, 'Naugatuck WV': 1, 'SouthRidge High School': 1, 'Pennsylvania, US': 1, 'Dexter, Maine USA': 1, 'louisiana': 1, 'rhode island': 1, 'Arizona, USA': 1, 'illinois, usa': 1, 'MA, USA': 1, 'Vermont USA': 1, 'Kodiak, Alaska, USA': 1, 'Ma': 1, 'Lithonia': 1, 'danville': 1, 'Chester VA': 1, 'Malverne NY': 1, 'Washington DC': 1, 'dothan': 1, 'fl': 1, 'virginia, u.s.a.': 1, '-': 1, 'Missouri, United States': 1, 'Mustang, OK': 1, 'IN': 1, 'New York, United States': 1, 'LAKE MARY, FL': 1, 'Wheeling, West V,irginia United States': 1, 'Waterford, MI': 1, 'california usa': 1, 'St.Pete Florida': 1, 'mo': 1, 'DILLSBURG PA': 1, 'ga/us': 1, 'Waynesboro VA': 1, 'South Carolina USA': 1, 'Troy, Montana': 1, 'South Dakota': 1, 'MI and TN': 1, 'Houston Texas': 1, 'Minnesota USA': 1, 'connecticut usa': 1, 'Cabot, Arkansas': 1, 'Springfield, MA': 1, 'Richmond VA': 1, 'Michigan, United States': 1, 'INDIANA': 1, 'virginia': 1, 'farmington. michigan': 1, 'oklahoma': 1, 'ft lauderdale': 1, 'New Zealand': 1, 'Dallas, Tx, USA': 1, 'Maryland USA': 1, 'CALIFORNIA': 1, 'Puerto Rico': 1, 'Pittsburgh': 1, 'philadelphia, pa': 1, 'lexington, sc': 1, 'Vermont': 1, 'North Carolina USA': 1, 'iowa': 1, 'United States of America': 1, 'georgia': 1, 'Michigan. Senior year in Indiana through correspondence (still lived in Michigan)': 1, 'Uncasville, CT': 1, 'NY USA': 1, 'New  York': 1, 'Muskegon Heights, Michigan': 1, 'Los Angeles, CA': 1, 'Army base Pisa Italy, Livorno Unit highschool': 1, 'Live Oak FL': 1, 'oh': 1, 'ndia': 1, 'Norway': 1, 'Queens, N.Y.': 1, 'seneca high': 1, 'Hoffman Estates, IL': 1, 'Eden, NC': 1, 'massachusetts': 1, 'Orange, CA': 1, 'HI': 1, 'meadowbrook highschool': 1, 'Orange, Tx': 1, 'Spring Mills Pa.': 1, 'San Diego, CA, USA': 1, 'Maryland/USA': 1, 'va': 1, 'Yakima Wa': 1, 'TEXAS': 1, 'USA, New Jersey': 1, 'Rochester, NY': 1, 'Paramus, NJ': 1, 'Florida ,USA': 1, 'Rock Island, IL': 1, 'Chesterville, Ontario, Canada': 1})
''')

st.write('''
* Are there any duplicated rows in the data?
    * No, the code and output below shows that no rows are dropped after duplicated rows are removed.
''')
st.code('''
df.shape == df.drop_duplicates().shape
''')
st.code('''
True
''')

st.write('''
* Can you normalize the data in the `state` column?
    * Yes, but it will take a little bit of work. I\'ll define a function `normalize_state`, which will take an input string from the `state` column and will output the 2-letter state
    abbreviation. The template of the function is shown below, where `pass` will need to be replaced appropriately.
''')
st.code('''
def normalize_state(state_string):
    pass
''')

st.write('''
The two dictionaries below, `abbrev_to_us_state` and `us_state_to_abbrev`, allow to move back and forth between the 2-letter abbreviation and the name of a state. Note that each dictionary
contains 57 key-value pairs, corrresponding to the 50 states, a federal district (Washington, D.C.), five major territories, and a key-pair `'NA': 'Unknown or foreign'` intended for foreign
locations or situations in which there is not enough info to identify the state. The function `normalize_state` will return a key from `abbrev_to_us_state`.
''')
st.code('''
abbrev_to_us_state = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'Unknown or foreign',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

us_state_to_abbrev = dict(map(reversed, abbrev_to_us_state.items()))
''')

abbrev_to_us_state = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'Unknown or foreign',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

us_state_to_abbrev = dict(map(reversed, abbrev_to_us_state.items()))

st.write('''
For future use, let's save the list of states and the list of abbreviations, both in upper case, into new variables.
''')
st.code('''
us_state_name_list = [state_name.upper() for state_name in list(abbrev_to_us_state.values())]
us_state_abbrev_list = [abbrev.upper() for abbrev in list(abbrev_to_us_state.keys())]
''')

us_state_name_list = [state_name.upper() for state_name in list(abbrev_to_us_state.values())]
us_state_abbrev_list = [abbrev.upper() for abbrev in list(abbrev_to_us_state.keys())]

st.write('''
Let\'s see next what values are in the `state` column. As shown in the output below, there are 363 unique values in the columns. These 363 values will need to be mapped down to
the 57 abbreviations. Note the variations in the values. For example, Georgia is indicated as `'GA'` (all uppercase letters), or `'ga'` (all lowercase letters), or `'Ga.'` (a
combination of uppercase, lowercase, and periods). Similarly, New York is indicated as `'NY'`, `'ny'`, or `'N.Y.'`.
''')
st.code('''
lst = np.sort(df.state.unique().astype(str))
print(lst.size, lst)
''')
st.code('''
363 ['-' 'AL' 'AR, USA' 'AZ' 'Alabama' 'Alabama, USA' 'Alaska' 'America'
 'Arizona' 'Arizona, USA' 'Arkansas' 'Arkansas, USA'
 'Army base Pisa Italy, Livorno Unit highschool' 'Aruba' 'Atlanta, GA'
 'Bristol, Connecticut' 'Buenos Aires, Argentina' 'CA' 'CA, USA'
 'CALIFORNIA' 'CO' 'CT' 'CT, USA' 'Cabot, Arkansas' 'Caflifornia'
 'California' 'California, USA' 'California, United States' 'Canada'
 'Carson City' 'Cathlamet, WA'
 'Cheshire High School, Cheshire, Connecticut' 'Chester VA'
 'Chesterville, Ontario, Canada' 'Chicago' 'Cincinnati Ohio'
 'Cincinnati, Ohio' 'Colorado' 'Columbia, SC' 'Columbus, Ohio'
 'Connecticut' 'Converse' 'DILLSBURG PA' 'Dallas, Tx, USA'
 'Dalton High, Dalton, Ga.' 'Decatur, AL' 'Delaware' 'Des Moines, Iowa'
 'Dexter, Maine USA' 'Duval' 'Eden, NC' 'FL' 'FLORIDA' 'FLORIDA, USA'
 'Florida' 'Florida ,USA' 'Florida, USA' 'Flushing, Michigan' 'Framingham'
 'Franklin, NC' 'GA' 'Georgia' 'Georgia, US' 'Georgia, USA' 'Germany' 'HI'
 'Hampshire High, Hampshire, Illinois' 'Hawaii' 'Henderson, NV'
 'Henrico, VA, USA' 'Hoffman Estates, IL' 'Houston Texas' 'IL' 'ILLINOIS'
 'IN' 'INDIANA' 'Idaho' 'Illinois' 'Illinois, USA' 'Independence, MO USA'
 'India' 'Indiana' 'Indiana, USA' 'Iowa' 'Iowa  United States'
 'Iowa, United States' 'KS, USA' 'KY' 'KY, USA' 'Kansas' 'Kentucky'
 'Kodiak, Alaska, USA' 'LA' 'LAKE MARY, FL' 'Lakeland FL' 'Lakewood, CO'
 'Lithonia' 'Live Oak FL' 'Livingston, Tx' 'Los Angeles, CA' 'Louisiana'
 'Louisiana, USA' 'Lousiana' 'MA' 'MA, USA' 'MI' 'MI and TN' 'MICHIGAN'
 'MN' 'MO' 'MO USA' 'Ma' 'Maine' 'Maine, USA' 'Malaysia' 'Malverne NY'
 'Maryland' 'Maryland USA' 'Maryland, USA' 'Maryland/USA' 'Massachusettes'
 'Massachusetts' 'Miami' 'Michigan' 'Michigan (US)' 'Michigan USA'
 'Michigan, US' 'Michigan, USA' 'Michigan, United States'
 'Michigan. Senior year in Indiana through correspondence (still lived in Michigan)'
 'Minnesota' 'Minnesota USA' 'Mississippi' 'Missouri' 'Missouri, USA'
 'Missouri, United States' 'Montana' 'Montgomery County, PA'
 'Muskegon Heights, Michigan' 'Mustang, OK' 'NC' 'NC, USA' 'NEW CASTLE'
 'NEW YORK' 'NH' 'NJ' 'NJ, USA' 'NORTH CAROLINA' 'NY' 'NY USA' 'NY, USA'
 'Naugatuck WV' 'Nebraska' 'Nevada' 'New  Jersey USA' 'New  York'
 'New Hampshire' 'New Jersey' 'New Mexico' 'New York' 'New York Ciry'
 'New York City' 'New York State' 'New York USA' 'New York, US'
 'New York, USA' 'New York, United States' 'New Zealand' 'Norfolk, VA'
 'North Carolina' 'North Carolina USA' 'North Carolina, USA'
 'North Carolina, United States' 'North Florida' 'Norway' 'OH' 'OKlahoma'
 'Ocala, FL' 'Ohio' 'Ohio, USA' 'Oklahoma' 'Orange County, California'
 'Orange, CA' 'Orange, Tx' 'Oregon' 'Oregon, USA' 'PA' 'PA, USA'
 'Paradise, California' 'Paramus, NJ' 'Pearl City, HI' 'Pennsylvania'
 'Pennsylvania, US' 'Pennsylvania, USA' 'Pennsylvania, United States'
 'Pennsyvania' 'Philippines' 'Pittsburgh' 'Pittsburgh, Pennsylvania'
 'Plant City, Florida' 'Puerto Rico' 'Quartz Hill' 'Queens, N.Y.' 'RI'
 'Rhode Island' 'Richmond VA' 'Rochester, NY' 'Rock Island, IL'
 'Rogers, Arkansas' 'SC' 'SC,  US' 'SD' 'SOUTH CAROLINA'
 'San Diego, CA, USA' 'Santa Clara' 'Singapore' 'Slovakia'
 'South Carolina' 'South Carolina USA' 'South Dakota'
 'SouthRidge High School' 'Spring Mills Pa.' 'Springfield, MA'
 'St. Thomas, USVI' 'St.Pete Florida' 'TEXAS' 'TN' 'TX' 'Tampa, FL'
 'Tennessee' 'Tennessee, USA' 'Texas' 'Texas, USA' 'Troy, Montana' 'US'
 'US, NY' 'USA' 'USA, New Jersey' 'Uncasville, CT' 'United States'
 'United States of America' 'United States, Illinois' 'United States, MN'
 'Utah' 'Utah, USA' 'Utah, United States of America' 'VA' 'VA, USA'
 'Vermont' 'Vermont USA' 'Vermont, United States' 'Virginia'
 'Virginia, USA' 'W.V' 'WA' 'WA State' 'WI' 'WI, USA' 'WISCONSIN' 'WV'
 'Walla Walla, WA' 'Washington' 'Washington DC' 'Washington State'
 'Washington State, USA' 'Waterford, MI' 'Waynesboro VA' 'West Virginia'
 'Westfield,NJ' 'Wheeling, West V,irginia United States' 'Whitwell, TN'
 'Wisconsin' 'Worthington' 'Yakima Wa' 'alabama' 'arizona'
 'boston massachusetts' 'californi' 'california' 'california usa' 'canada'
 'colorado' 'connecticut' 'connecticut usa' 'danville' 'denver, colorado'
 'dothan' 'el paso, tx' 'farmington. michigan' 'fl' 'florida'
 'ft lauderdale' 'ga' 'ga. usa' 'ga/us' 'georgia'
 'grosse pointe, michigan, united states' 'hawaii' 'idaho' 'idaho, usa'
 'illinois' 'illinois, usa' 'indiana' 'iowa' 'kansas' 'kentucky'
 'lake butler florida' 'lexington, sc' 'los angeles, ca' 'louisiana'
 'louisville ky' 'ma usa' 'maine' 'maryland' 'maryland, usa'
 'massachusetts' 'md' 'meadowbrook highschool' 'michigan' 'michigan, usa'
 'millinocket' 'minnesota' 'missouri' 'mo' 'modern college,India'
 'multiple locations' 'nc' 'ndia' 'nebraska' 'new jersey' 'new york' 'nj'
 'north carolina' 'ny' 'nyc, ny' 'oh' 'ohio' 'oklahoma' 'pennsylvania'
 'philadelphia, pa' 'port huron, michigan' 'rhode island' 'richmond, IN'
 'sc' 'seneca high' 'sheboygan, wi' 'tennesee' 'tennessee'
 'tennessee, usa' 'texas' 'texas,usa' 'tx' 'united states' 'usa' 'usa Fla'
 'utah' 'va' 'va, usa' 'virginia' 'virginia, u.s.a.' 'washington'
 'wisconsin' 'youngstown oh']
''')

st.write('''
To reduce the number of unique values, we can delete all periods and change all letters to uppercase. The number of unique values
dropped from 363 to 294, but there is more work to be done.
''')
st.code('''
def apply_uppercase_and_delete_all_periods(string_value):
    temp = str.replace(string_value, ".", "")
    temp = temp.upper()
    return temp
lst_1 = np.stack(np.vectorize(apply_uppercase_and_delete_all_periods)(lst), axis=0)
lst_2 = np.sort(np.unique(lst_1))
print(lst_2.size, lst_2)
''')
st.code('''
294 ['-' 'AL' 'ALABAMA' 'ALABAMA, USA' 'ALASKA' 'AMERICA' 'AR, USA' 'ARIZONA'
 'ARIZONA, USA' 'ARKANSAS' 'ARKANSAS, USA'
 'ARMY BASE PISA ITALY, LIVORNO UNIT HIGHSCHOOL' 'ARUBA' 'ATLANTA, GA'
 'AZ' 'BOSTON MASSACHUSETTS' 'BRISTOL, CONNECTICUT'
 'BUENOS AIRES, ARGENTINA' 'CA' 'CA, USA' 'CABOT, ARKANSAS' 'CAFLIFORNIA'
 'CALIFORNI' 'CALIFORNIA' 'CALIFORNIA USA' 'CALIFORNIA, UNITED STATES'
 'CALIFORNIA, USA' 'CANADA' 'CARSON CITY' 'CATHLAMET, WA'
 'CHESHIRE HIGH SCHOOL, CHESHIRE, CONNECTICUT' 'CHESTER VA'
 'CHESTERVILLE, ONTARIO, CANADA' 'CHICAGO' 'CINCINNATI OHIO'
 'CINCINNATI, OHIO' 'CO' 'COLORADO' 'COLUMBIA, SC' 'COLUMBUS, OHIO'
 'CONNECTICUT' 'CONNECTICUT USA' 'CONVERSE' 'CT' 'CT, USA'
 'DALLAS, TX, USA' 'DALTON HIGH, DALTON, GA' 'DANVILLE' 'DECATUR, AL'
 'DELAWARE' 'DENVER, COLORADO' 'DES MOINES, IOWA' 'DEXTER, MAINE USA'
 'DILLSBURG PA' 'DOTHAN' 'DUVAL' 'EDEN, NC' 'EL PASO, TX'
 'FARMINGTON MICHIGAN' 'FL' 'FLORIDA' 'FLORIDA ,USA' 'FLORIDA, USA'
 'FLUSHING, MICHIGAN' 'FRAMINGHAM' 'FRANKLIN, NC' 'FT LAUDERDALE' 'GA'
 'GA USA' 'GA/US' 'GEORGIA' 'GEORGIA, US' 'GEORGIA, USA' 'GERMANY'
 'GROSSE POINTE, MICHIGAN, UNITED STATES'
 'HAMPSHIRE HIGH, HAMPSHIRE, ILLINOIS' 'HAWAII' 'HENDERSON, NV'
 'HENRICO, VA, USA' 'HI' 'HOFFMAN ESTATES, IL' 'HOUSTON TEXAS' 'IDAHO'
 'IDAHO, USA' 'IL' 'ILLINOIS' 'ILLINOIS, USA' 'IN' 'INDEPENDENCE, MO USA'
 'INDIA' 'INDIANA' 'INDIANA, USA' 'IOWA' 'IOWA  UNITED STATES'
 'IOWA, UNITED STATES' 'KANSAS' 'KENTUCKY' 'KODIAK, ALASKA, USA' 'KS, USA'
 'KY' 'KY, USA' 'LA' 'LAKE BUTLER FLORIDA' 'LAKE MARY, FL' 'LAKELAND FL'
 'LAKEWOOD, CO' 'LEXINGTON, SC' 'LITHONIA' 'LIVE OAK FL' 'LIVINGSTON, TX'
 'LOS ANGELES, CA' 'LOUISIANA' 'LOUISIANA, USA' 'LOUISVILLE KY' 'LOUSIANA'
 'MA' 'MA USA' 'MA, USA' 'MAINE' 'MAINE, USA' 'MALAYSIA' 'MALVERNE NY'
 'MARYLAND' 'MARYLAND USA' 'MARYLAND, USA' 'MARYLAND/USA' 'MASSACHUSETTES'
 'MASSACHUSETTS' 'MD' 'MEADOWBROOK HIGHSCHOOL' 'MI' 'MI AND TN' 'MIAMI'
 'MICHIGAN' 'MICHIGAN (US)'
 'MICHIGAN SENIOR YEAR IN INDIANA THROUGH CORRESPONDENCE (STILL LIVED IN MICHIGAN)'
 'MICHIGAN USA' 'MICHIGAN, UNITED STATES' 'MICHIGAN, US' 'MICHIGAN, USA'
 'MILLINOCKET' 'MINNESOTA' 'MINNESOTA USA' 'MISSISSIPPI' 'MISSOURI'
 'MISSOURI, UNITED STATES' 'MISSOURI, USA' 'MN' 'MO' 'MO USA'
 'MODERN COLLEGE,INDIA' 'MONTANA' 'MONTGOMERY COUNTY, PA'
 'MULTIPLE LOCATIONS' 'MUSKEGON HEIGHTS, MICHIGAN' 'MUSTANG, OK'
 'NAUGATUCK WV' 'NC' 'NC, USA' 'NDIA' 'NEBRASKA' 'NEVADA'
 'NEW  JERSEY USA' 'NEW  YORK' 'NEW CASTLE' 'NEW HAMPSHIRE' 'NEW JERSEY'
 'NEW MEXICO' 'NEW YORK' 'NEW YORK CIRY' 'NEW YORK CITY' 'NEW YORK STATE'
 'NEW YORK USA' 'NEW YORK, UNITED STATES' 'NEW YORK, US' 'NEW YORK, USA'
 'NEW ZEALAND' 'NH' 'NJ' 'NJ, USA' 'NORFOLK, VA' 'NORTH CAROLINA'
 'NORTH CAROLINA USA' 'NORTH CAROLINA, UNITED STATES'
 'NORTH CAROLINA, USA' 'NORTH FLORIDA' 'NORWAY' 'NY' 'NY USA' 'NY, USA'
 'NYC, NY' 'OCALA, FL' 'OH' 'OHIO' 'OHIO, USA' 'OKLAHOMA'
 'ORANGE COUNTY, CALIFORNIA' 'ORANGE, CA' 'ORANGE, TX' 'OREGON'
 'OREGON, USA' 'PA' 'PA, USA' 'PARADISE, CALIFORNIA' 'PARAMUS, NJ'
 'PEARL CITY, HI' 'PENNSYLVANIA' 'PENNSYLVANIA, UNITED STATES'
 'PENNSYLVANIA, US' 'PENNSYLVANIA, USA' 'PENNSYVANIA' 'PHILADELPHIA, PA'
 'PHILIPPINES' 'PITTSBURGH' 'PITTSBURGH, PENNSYLVANIA'
 'PLANT CITY, FLORIDA' 'PORT HURON, MICHIGAN' 'PUERTO RICO' 'QUARTZ HILL'
 'QUEENS, NY' 'RHODE ISLAND' 'RI' 'RICHMOND VA' 'RICHMOND, IN'
 'ROCHESTER, NY' 'ROCK ISLAND, IL' 'ROGERS, ARKANSAS' 'SAN DIEGO, CA, USA'
 'SANTA CLARA' 'SC' 'SC,  US' 'SD' 'SENECA HIGH' 'SHEBOYGAN, WI'
 'SINGAPORE' 'SLOVAKIA' 'SOUTH CAROLINA' 'SOUTH CAROLINA USA'
 'SOUTH DAKOTA' 'SOUTHRIDGE HIGH SCHOOL' 'SPRING MILLS PA'
 'SPRINGFIELD, MA' 'ST THOMAS, USVI' 'STPETE FLORIDA' 'TAMPA, FL'
 'TENNESEE' 'TENNESSEE' 'TENNESSEE, USA' 'TEXAS' 'TEXAS, USA' 'TEXAS,USA'
 'TN' 'TROY, MONTANA' 'TX' 'UNCASVILLE, CT' 'UNITED STATES'
 'UNITED STATES OF AMERICA' 'UNITED STATES, ILLINOIS' 'UNITED STATES, MN'
 'US' 'US, NY' 'USA' 'USA FLA' 'USA, NEW JERSEY' 'UTAH'
 'UTAH, UNITED STATES OF AMERICA' 'UTAH, USA' 'VA' 'VA, USA' 'VERMONT'
 'VERMONT USA' 'VERMONT, UNITED STATES' 'VIRGINIA' 'VIRGINIA, USA' 'WA'
 'WA STATE' 'WALLA WALLA, WA' 'WASHINGTON' 'WASHINGTON DC'
 'WASHINGTON STATE' 'WASHINGTON STATE, USA' 'WATERFORD, MI'
 'WAYNESBORO VA' 'WEST VIRGINIA' 'WESTFIELD,NJ'
 'WHEELING, WEST V,IRGINIA UNITED STATES' 'WHITWELL, TN' 'WI' 'WI, USA'
 'WISCONSIN' 'WORTHINGTON' 'WV' 'YAKIMA WA' 'YOUNGSTOWN OH']
''')

st.write('''
To match an input string from the `state` column with a 2-letter state abbreviation, I'll use the `fuzzywuzzy` package, which gives
a numerical measure of the similarity between two strings. In the code below, you can see that the largest number corresponds to the
match to `WEST VIRIGINIA`.
''')
st.code('''
string_value = 'WHEELING, WEST V,IRGINIA UNITED STATES'
for potential_match_state in us_state_name_list:
    print(f"{string_value} -> {potential_match_state} = {fuzz.ratio(string_value, potential_match_state)}")
''')
st.code('''
WHEELING, WEST V,IRGINIA UNITED STATES -> ALASKA = 14
WHEELING, WEST V,IRGINIA UNITED STATES -> ALABAMA = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> ARKANSAS = 17
WHEELING, WEST V,IRGINIA UNITED STATES -> AMERICAN SAMOA = 23
WHEELING, WEST V,IRGINIA UNITED STATES -> ARIZONA = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> CALIFORNIA = 25
WHEELING, WEST V,IRGINIA UNITED STATES -> COLORADO = 17
WHEELING, WEST V,IRGINIA UNITED STATES -> CONNECTICUT = 12
WHEELING, WEST V,IRGINIA UNITED STATES -> DISTRICT OF COLUMBIA = 21
WHEELING, WEST V,IRGINIA UNITED STATES -> DELAWARE = 17
WHEELING, WEST V,IRGINIA UNITED STATES -> FLORIDA = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> GEORGIA = 22
WHEELING, WEST V,IRGINIA UNITED STATES -> GUAM = 10
WHEELING, WEST V,IRGINIA UNITED STATES -> HAWAII = 14
WHEELING, WEST V,IRGINIA UNITED STATES -> IOWA = 10
WHEELING, WEST V,IRGINIA UNITED STATES -> IDAHO = 5
WHEELING, WEST V,IRGINIA UNITED STATES -> ILLINOIS = 17
WHEELING, WEST V,IRGINIA UNITED STATES -> INDIANA = 27
WHEELING, WEST V,IRGINIA UNITED STATES -> KANSAS = 18
WHEELING, WEST V,IRGINIA UNITED STATES -> KENTUCKY = 17
WHEELING, WEST V,IRGINIA UNITED STATES -> LOUISIANA = 30
WHEELING, WEST V,IRGINIA UNITED STATES -> MASSACHUSETTS = 12
WHEELING, WEST V,IRGINIA UNITED STATES -> MARYLAND = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> MAINE = 14
WHEELING, WEST V,IRGINIA UNITED STATES -> MICHIGAN = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> MINNESOTA = 26
WHEELING, WEST V,IRGINIA UNITED STATES -> MISSOURI = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> NORTHERN MARIANA ISLANDS = 35
WHEELING, WEST V,IRGINIA UNITED STATES -> MISSISSIPPI = 20
WHEELING, WEST V,IRGINIA UNITED STATES -> MONTANA = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> UNKNOWN OR FOREIGN = 11
WHEELING, WEST V,IRGINIA UNITED STATES -> NORTH CAROLINA = 19
WHEELING, WEST V,IRGINIA UNITED STATES -> NORTH DAKOTA = 20
WHEELING, WEST V,IRGINIA UNITED STATES -> NEBRASKA = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> NEW HAMPSHIRE = 24
WHEELING, WEST V,IRGINIA UNITED STATES -> NEW JERSEY = 12
WHEELING, WEST V,IRGINIA UNITED STATES -> NEW MEXICO = 12
WHEELING, WEST V,IRGINIA UNITED STATES -> NEVADA = 23
WHEELING, WEST V,IRGINIA UNITED STATES -> NEW YORK = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> OHIO = 10
WHEELING, WEST V,IRGINIA UNITED STATES -> OKLAHOMA = 9
WHEELING, WEST V,IRGINIA UNITED STATES -> OREGON = 9
WHEELING, WEST V,IRGINIA UNITED STATES -> PENNSYLVANIA = 24
WHEELING, WEST V,IRGINIA UNITED STATES -> PUERTO RICO = 8
WHEELING, WEST V,IRGINIA UNITED STATES -> RHODE ISLAND = 20
WHEELING, WEST V,IRGINIA UNITED STATES -> SOUTH CAROLINA = 19
WHEELING, WEST V,IRGINIA UNITED STATES -> SOUTH DAKOTA = 20
WHEELING, WEST V,IRGINIA UNITED STATES -> TENNESSEE = 9
WHEELING, WEST V,IRGINIA UNITED STATES -> TEXAS = 14
WHEELING, WEST V,IRGINIA UNITED STATES -> UTAH = 14
WHEELING, WEST V,IRGINIA UNITED STATES -> VIRGINIA = 35
WHEELING, WEST V,IRGINIA UNITED STATES -> VIRGIN ISLANDS = 42
WHEELING, WEST V,IRGINIA UNITED STATES -> VERMONT = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> WASHINGTON = 29
WHEELING, WEST V,IRGINIA UNITED STATES -> WISCONSIN = 13
WHEELING, WEST V,IRGINIA UNITED STATES -> WEST VIRGINIA = 51
WHEELING, WEST V,IRGINIA UNITED STATES -> WYOMING = 18
''')

st.write('''
The `fuzzywuzzy` package has a function that automatically identifies the best match in a list, as shown below.
''')
st.code('''
best_match = process.extractOne(string_value, us_state_name_list)
print(f"{string_value} -> {best_match}")
''')
st.code('''
WHEELING, WEST V,IRGINIA UNITED STATES -> ('WEST VIRGINIA', 86)
''')

st.write('''
The code block below identifies the bext match for each string input. You'll notice several issues:

1. Some strings include the abbreviation while others include the actual name and sometimes the two ways of describing a state gets
matched to two different states. For example, `AZ` is matched to `FLORIDA`, while `ARIZONA` is matched to `ARIZONA`. To improve the match,
I'll replace the abbreviation with the actual name.
2. The string `AMERICA` is matched with `AMERICAN SAMOA`. The match will imrpove if we remove the substring `AMERICA` from any string input and
replace it with `-`.
3. The string `ST THOMAS, USVI` is matched with `OKLAHOMA`. The match will improve if insert a space before `VI` and then replace the `VI`
abbrviation with the actual name of the territory.
''')
st.code('''
for string_value in lst_2:
    best_match = process.extractOne(string_value, us_state_name_list)
    print(f"{string_value} -> {best_match}")
''')
st.code('''
- -> ('ALASKA', 0)
AL -> ('ALASKA', 90)
ALABAMA -> ('ALABAMA', 100)
ALABAMA, USA -> ('ALABAMA', 90)
ALASKA -> ('ALASKA', 100)
AMERICA -> ('AMERICAN SAMOA', 90)
AR, USA -> ('AMERICAN SAMOA', 57)
ARIZONA -> ('ARIZONA', 100)
ARIZONA, USA -> ('ARIZONA', 90)
ARKANSAS -> ('ARKANSAS', 100)
ARKANSAS, USA -> ('ARKANSAS', 90)
ARMY BASE PISA ITALY, LIVORNO UNIT HIGHSCHOOL -> ('LOUISIANA', 48)
ARUBA -> ('ARKANSAS', 54)
ATLANTA, GA -> ('GUAM', 60)
AZ -> ('FLORIDA', 60)
BOSTON MASSACHUSETTS -> ('MASSACHUSETTS', 90)
BRISTOL, CONNECTICUT -> ('CONNECTICUT', 90)
BUENOS AIRES, ARGENTINA -> ('VIRGINIA', 56)
CA -> ('AMERICAN SAMOA', 90)
CA, USA -> ('AMERICAN SAMOA', 71)
CABOT, ARKANSAS -> ('ARKANSAS', 90)
CAFLIFORNIA -> ('CALIFORNIA', 95)
CALIFORNI -> ('CALIFORNIA', 95)
CALIFORNIA -> ('CALIFORNIA', 100)
CALIFORNIA USA -> ('CALIFORNIA', 95)
CALIFORNIA, UNITED STATES -> ('CALIFORNIA', 90)
CALIFORNIA, USA -> ('CALIFORNIA', 90)
CANADA -> ('NEVADA', 67)
CARSON CITY -> ('CONNECTICUT', 55)
CATHLAMET, WA -> ('ALABAMA', 51)
CHESHIRE HIGH SCHOOL, CHESHIRE, CONNECTICUT -> ('CONNECTICUT', 90)
CHESTER VA -> ('WEST VIRGINIA', 52)
CHESTERVILLE, ONTARIO, CANADA -> ('ARIZONA', 62)
CHICAGO -> ('OHIO', 55)
CINCINNATI OHIO -> ('OHIO', 90)
CINCINNATI, OHIO -> ('OHIO', 90)
CO -> ('COLORADO', 90)
COLORADO -> ('COLORADO', 100)
COLUMBIA, SC -> ('DISTRICT OF COLUMBIA', 86)
COLUMBUS, OHIO -> ('OHIO', 90)
CONNECTICUT -> ('CONNECTICUT', 100)
CONNECTICUT USA -> ('CONNECTICUT', 95)
CONVERSE -> ('NEW JERSEY', 56)
CT -> ('CONNECTICUT', 90)
CT, USA -> ('AMERICAN SAMOA', 57)
DALLAS, TX, USA -> ('ALASKA', 60)
DALTON HIGH, DALTON, GA -> ('IDAHO', 54)
DANVILLE -> ('MAINE', 46)
DECATUR, AL -> ('ALASKA', 45)
DELAWARE -> ('DELAWARE', 100)
DENVER, COLORADO -> ('COLORADO', 90)
DES MOINES, IOWA -> ('IOWA', 90)
DEXTER, MAINE USA -> ('MAINE', 90)
DILLSBURG PA -> ('GUAM', 45)
DOTHAN -> ('MONTANA', 62)
DUVAL -> ('GUAM', 44)
EDEN, NC -> ('NEW JERSEY', 45)
EL PASO, TX -> ('ALASKA', 45)
FARMINGTON MICHIGAN -> ('MICHIGAN', 90)
FL -> ('FLORIDA', 90)
FLORIDA -> ('FLORIDA', 100)
FLORIDA ,USA -> ('FLORIDA', 90)
FLORIDA, USA -> ('FLORIDA', 90)
FLUSHING, MICHIGAN -> ('MICHIGAN', 90)
FRAMINGHAM -> ('GUAM', 68)
FRANKLIN, NC -> ('NORTH CAROLINA', 46)
FT LAUDERDALE -> ('FLORIDA', 50)
GA -> ('MICHIGAN', 90)
GA USA -> ('AMERICAN SAMOA', 60)
GA/US -> ('AMERICAN SAMOA', 54)
GEORGIA -> ('GEORGIA', 100)
GEORGIA, US -> ('GEORGIA', 90)
GEORGIA, USA -> ('GEORGIA', 90)
GERMANY -> ('GEORGIA', 57)
GROSSE POINTE, MICHIGAN, UNITED STATES -> ('MICHIGAN', 90)
HAMPSHIRE HIGH, HAMPSHIRE, ILLINOIS -> ('ILLINOIS', 90)
HAWAII -> ('HAWAII', 100)
HENDERSON, NV -> ('VERMONT', 51)
HENRICO, VA, USA -> ('AMERICAN SAMOA', 54)
HI -> ('MICHIGAN', 90)
HOFFMAN ESTATES, IL -> ('MAINE', 72)
HOUSTON TEXAS -> ('TEXAS', 90)
IDAHO -> ('IDAHO', 100)
IDAHO, USA -> ('IDAHO', 90)
IL -> ('ILLINOIS', 90)
ILLINOIS -> ('ILLINOIS', 100)
ILLINOIS, USA -> ('ILLINOIS', 90)
IN -> ('ILLINOIS', 90)
INDEPENDENCE, MO USA -> ('MAINE', 54)
INDIA -> ('INDIANA', 83)
INDIANA -> ('INDIANA', 100)
INDIANA, USA -> ('INDIANA', 90)
IOWA -> ('IOWA', 100)
IOWA  UNITED STATES -> ('IOWA', 90)
IOWA, UNITED STATES -> ('IOWA', 90)
KANSAS -> ('KANSAS', 100)
KENTUCKY -> ('KENTUCKY', 100)
KODIAK, ALASKA, USA -> ('ALASKA', 90)
KS, USA -> ('GUAM', 51)
KY -> ('KENTUCKY', 90)
KY, USA -> ('GUAM', 51)
LA -> ('ALASKA', 90)
LAKE BUTLER FLORIDA -> ('FLORIDA', 90)
LAKE MARY, FL -> ('MARYLAND', 57)
LAKELAND FL -> ('ALASKA', 60)
LAKEWOOD, CO -> ('NEW MEXICO', 46)
LEXINGTON, SC -> ('WASHINGTON', 52)
LITHONIA -> ('CALIFORNIA', 67)
LIVE OAK FL -> ('IOWA', 45)
LIVINGSTON, TX -> ('WASHINGTON', 50)
LOS ANGELES, CA -> ('MAINE', 54)
LOUISIANA -> ('LOUISIANA', 100)
LOUISIANA, USA -> ('LOUISIANA', 90)
LOUISVILLE KY -> ('LOUISIANA', 55)
LOUSIANA -> ('LOUISIANA', 94)
MA -> ('ALABAMA', 90)
MA USA -> ('AMERICAN SAMOA', 60)
MA, USA -> ('AMERICAN SAMOA', 57)
MAINE -> ('MAINE', 100)
MAINE, USA -> ('MAINE', 90)
MALAYSIA -> ('ALASKA', 71)
MALVERNE NY -> ('MAINE', 54)
MARYLAND -> ('MARYLAND', 100)
MARYLAND USA -> ('MARYLAND', 90)
MARYLAND, USA -> ('MARYLAND', 90)
MARYLAND/USA -> ('MARYLAND', 90)
MASSACHUSETTES -> ('MASSACHUSETTS', 96)
MASSACHUSETTS -> ('MASSACHUSETTS', 100)
MD -> ('GUAM', 60)
MEADOWBROOK HIGHSCHOOL -> ('COLORADO', 53)
MI -> ('MICHIGAN', 90)
MI AND TN -> ('NORTHERN MARIANA ISLANDS', 57)
MIAMI -> ('DISTRICT OF COLUMBIA', 60)
MICHIGAN -> ('MICHIGAN', 100)
MICHIGAN (US) -> ('MICHIGAN', 90)
MICHIGAN SENIOR YEAR IN INDIANA THROUGH CORRESPONDENCE (STILL LIVED IN MICHIGAN) -> ('INDIANA', 60)
MICHIGAN USA -> ('MICHIGAN', 90)
MICHIGAN, UNITED STATES -> ('MICHIGAN', 90)
MICHIGAN, US -> ('MICHIGAN', 90)
MICHIGAN, USA -> ('MICHIGAN', 90)
MILLINOCKET -> ('ILLINOIS', 63)
MINNESOTA -> ('MINNESOTA', 100)
MINNESOTA USA -> ('MINNESOTA', 95)
MISSISSIPPI -> ('MISSISSIPPI', 100)
MISSOURI -> ('MISSOURI', 100)
MISSOURI, UNITED STATES -> ('MISSOURI', 90)
MISSOURI, USA -> ('MISSOURI', 90)
MN -> ('GUAM', 60)
MO -> ('AMERICAN SAMOA', 90)
MO USA -> ('AMERICAN SAMOA', 60)
MODERN COLLEGE,INDIA -> ('INDIANA', 75)
MONTANA -> ('MONTANA', 100)
MONTGOMERY COUNTY, PA -> ('MONTANA', 51)
MULTIPLE LOCATIONS -> ('ARIZONA', 56)
MUSKEGON HEIGHTS, MICHIGAN -> ('MICHIGAN', 90)
MUSTANG, OK -> ('UTAH', 68)
NAUGATUCK WV -> ('KENTUCKY', 50)
NC -> ('MICHIGAN', 60)
NC, USA -> ('AMERICAN SAMOA', 57)
NDIA -> ('INDIANA', 90)
NEBRASKA -> ('NEBRASKA', 100)
NEVADA -> ('NEVADA', 100)
NEW  JERSEY USA -> ('NEW JERSEY', 86)
NEW  YORK -> ('NEW YORK', 95)
NEW CASTLE -> ('NEW HAMPSHIRE', 61)
NEW HAMPSHIRE -> ('NEW HAMPSHIRE', 100)
NEW JERSEY -> ('NEW JERSEY', 100)
NEW MEXICO -> ('NEW MEXICO', 100)
NEW YORK -> ('NEW YORK', 100)
NEW YORK CIRY -> ('NEW YORK', 90)
NEW YORK CITY -> ('NEW YORK', 90)
NEW YORK STATE -> ('NEW YORK', 90)
NEW YORK USA -> ('NEW YORK', 90)
NEW YORK, UNITED STATES -> ('NEW YORK', 90)
NEW YORK, US -> ('NEW YORK', 90)
NEW YORK, USA -> ('NEW YORK', 90)
NEW ZEALAND -> ('MARYLAND', 53)
NH -> ('MICHIGAN', 60)
NJ -> ('MICHIGAN', 60)
NJ, USA -> ('AMERICAN SAMOA', 57)
NORFOLK, VA -> ('NORTH CAROLINA', 48)
NORTH CAROLINA -> ('NORTH CAROLINA', 100)
NORTH CAROLINA USA -> ('NORTH CAROLINA', 95)
NORTH CAROLINA, UNITED STATES -> ('NORTH CAROLINA', 90)
NORTH CAROLINA, USA -> ('NORTH CAROLINA', 95)
NORTH FLORIDA -> ('FLORIDA', 90)
NORWAY -> ('IOWA', 68)
NY -> ('MICHIGAN', 60)
NY USA -> ('AMERICAN SAMOA', 60)
NY, USA -> ('AMERICAN SAMOA', 57)
NYC, NY -> ('CONNECTICUT', 43)
OCALA, FL -> ('ALASKA', 57)
OH -> ('OHIO', 90)
OHIO -> ('OHIO', 100)
OHIO, USA -> ('OHIO', 90)
OKLAHOMA -> ('OKLAHOMA', 100)
ORANGE COUNTY, CALIFORNIA -> ('CALIFORNIA', 90)
ORANGE, CA -> ('MAINE', 54)
ORANGE, TX -> ('MAINE', 54)
OREGON -> ('OREGON', 100)
OREGON, USA -> ('OREGON', 90)
PA -> ('ALASKA', 45)
PA, USA -> ('AMERICAN SAMOA', 57)
PARADISE, CALIFORNIA -> ('CALIFORNIA', 90)
PARAMUS, NJ -> ('ALASKA', 45)
PEARL CITY, HI -> ('OHIO', 51)
PENNSYLVANIA -> ('PENNSYLVANIA', 100)
PENNSYLVANIA, UNITED STATES -> ('PENNSYLVANIA', 90)
PENNSYLVANIA, US -> ('PENNSYLVANIA', 95)
PENNSYLVANIA, USA -> ('PENNSYLVANIA', 95)
PENNSYVANIA -> ('PENNSYLVANIA', 96)
PHILADELPHIA, PA -> ('DELAWARE', 45)
PHILIPPINES -> ('MAINE', 54)
PITTSBURGH -> ('UTAH', 45)
PITTSBURGH, PENNSYLVANIA -> ('PENNSYLVANIA', 90)
PLANT CITY, FLORIDA -> ('FLORIDA', 90)
PORT HURON, MICHIGAN -> ('MICHIGAN', 90)
PUERTO RICO -> ('PUERTO RICO', 100)
QUARTZ HILL -> ('ARIZONA', 47)
QUEENS, NY -> ('NEW JERSEY', 40)
RHODE ISLAND -> ('RHODE ISLAND', 100)
RI -> ('AMERICAN SAMOA', 90)
RICHMOND VA -> ('ARIZONA', 56)
RICHMOND, IN -> ('RHODE ISLAND', 58)
ROCHESTER, NY -> ('OHIO', 45)
ROCK ISLAND, IL -> ('RHODE ISLAND', 67)
ROGERS, ARKANSAS -> ('ARKANSAS', 90)
SAN DIEGO, CA, USA -> ('KANSAS', 57)
SANTA CLARA -> ('ALASKA', 57)
SC -> ('WISCONSIN', 90)
SC,  US -> ('MASSACHUSETTS', 51)
SD -> ('ILLINOIS', 60)
SENECA HIGH -> ('NEVADA', 51)
SHEBOYGAN, WI -> ('HAWAII', 47)
SINGAPORE -> ('OREGON', 60)
SLOVAKIA -> ('PENNSYLVANIA', 68)
SOUTH CAROLINA -> ('SOUTH CAROLINA', 100)
SOUTH CAROLINA USA -> ('SOUTH CAROLINA', 95)
SOUTH DAKOTA -> ('SOUTH DAKOTA', 100)
SOUTHRIDGE HIGH SCHOOL -> ('UTAH', 68)
SPRING MILLS PA -> ('ARIZONA', 49)
SPRINGFIELD, MA -> ('MAINE', 51)
ST THOMAS, USVI -> ('OKLAHOMA', 45)
STPETE FLORIDA -> ('FLORIDA', 90)
TAMPA, FL -> ('ALABAMA', 50)
TENNESEE -> ('TENNESSEE', 94)
TENNESSEE -> ('TENNESSEE', 100)
TENNESSEE, USA -> ('TENNESSEE', 90)
TEXAS -> ('TEXAS', 100)
TEXAS, USA -> ('TEXAS', 90)
TEXAS,USA -> ('TEXAS', 90)
TN -> ('VERMONT', 60)
TROY, MONTANA -> ('MONTANA', 90)
TX -> ('VERMONT', 60)
UNCASVILLE, CT -> ('GUAM', 45)
UNITED STATES -> ('TEXAS', 54)
UNITED STATES OF AMERICA -> ('AMERICAN SAMOA', 61)
UNITED STATES, ILLINOIS -> ('ILLINOIS', 90)
UNITED STATES, MN -> ('TEXAS', 54)
US -> ('MASSACHUSETTS', 90)
US, NY -> ('AMERICAN SAMOA', 51)
USA -> ('ALASKA', 60)
USA FLA -> ('ALASKA', 59)
USA, NEW JERSEY -> ('NEW JERSEY', 90)
UTAH -> ('UTAH', 100)
UTAH, UNITED STATES OF AMERICA -> ('UTAH', 90)
UTAH, USA -> ('UTAH', 90)
VA -> ('NEVADA', 90)
VA, USA -> ('AMERICAN SAMOA', 51)
VERMONT -> ('VERMONT', 100)
VERMONT USA -> ('VERMONT', 90)
VERMONT, UNITED STATES -> ('VERMONT', 90)
VIRGINIA -> ('VIRGINIA', 100)
VIRGINIA, USA -> ('VIRGINIA', 90)
WA -> ('DELAWARE', 90)
WA STATE -> ('TEXAS', 51)
WALLA WALLA, WA -> ('ALASKA', 51)
WASHINGTON -> ('WASHINGTON', 100)
WASHINGTON DC -> ('WASHINGTON', 95)
WASHINGTON STATE -> ('WASHINGTON', 90)
WASHINGTON STATE, USA -> ('WASHINGTON', 90)
WATERFORD, MI -> ('IOWA', 64)
WAYNESBORO VA -> ('OREGON', 57)
WEST VIRGINIA -> ('WEST VIRGINIA', 100)
WESTFIELD,NJ -> ('WEST VIRGINIA', 48)
WHEELING, WEST V,IRGINIA UNITED STATES -> ('WEST VIRGINIA', 86)
WHITWELL, TN -> ('IOWA', 45)
WI -> ('WISCONSIN', 90)
WI, USA -> ('AMERICAN SAMOA', 51)
WISCONSIN -> ('WISCONSIN', 100)
WORTHINGTON -> ('WASHINGTON', 76)
WV -> ('NEW HAMPSHIRE', 57)
YAKIMA WA -> ('HAWAII', 54)
YOUNGSTOWN OH -> ('OHIO', 60)
''')

st.write('''
The code block below, makes the three changes described before. A quick look over the output identifies 86 as a
threshold value for a good match---values greater than or equal to 86 generate good matches in the sample, while
values lower than 86 generate bad matches.
''')
st.code('''
import re
new_states_list = []
for string_value in lst_2:
    temp_list = string_value.replace("USVI", "US VI")
    temp_list = re.split("[/, ]", temp_list)
    new_list = []
    for word in temp_list:
        if word in us_state_abbrev_list:
            new_list.append(str.upper(abbrev_to_us_state[word]))
        elif word == 'AMERICA':
            new_list.append('-')
        else:
            new_list.append(word)
    new_states_list.append(str.strip(' '.join(new_list)))

for string_value in new_states_list:
    best_match = process.extractOne(string_value, us_state_name_list)
    print(f"{string_value} -> {best_match}")
''')
st.code('''
- -> ('ALASKA', 0)
ALABAMA -> ('ALABAMA', 100)
ALABAMA -> ('ALABAMA', 100)
ALABAMA  USA -> ('ALABAMA', 90)
ALASKA -> ('ALASKA', 100)
- -> ('ALASKA', 0)
ARKANSAS  USA -> ('ARKANSAS', 90)
ARIZONA -> ('ARIZONA', 100)
ARIZONA  USA -> ('ARIZONA', 90)
ARKANSAS -> ('ARKANSAS', 100)
ARKANSAS  USA -> ('ARKANSAS', 90)
ARMY BASE PISA ITALY  LIVORNO UNIT HIGHSCHOOL -> ('LOUISIANA', 48)
ARUBA -> ('ARKANSAS', 54)
ATLANTA  GEORGIA -> ('GEORGIA', 90)
ARIZONA -> ('ARIZONA', 100)
BOSTON MASSACHUSETTS -> ('MASSACHUSETTS', 90)
BRISTOL  CONNECTICUT -> ('CONNECTICUT', 90)
BUENOS AIRES  ARGENTINA -> ('VIRGINIA', 56)
CALIFORNIA -> ('CALIFORNIA', 100)
CALIFORNIA  USA -> ('CALIFORNIA', 90)
CABOT  ARKANSAS -> ('ARKANSAS', 90)
CAFLIFORNIA -> ('CALIFORNIA', 95)
CALIFORNI -> ('CALIFORNIA', 95)
CALIFORNIA -> ('CALIFORNIA', 100)
CALIFORNIA USA -> ('CALIFORNIA', 95)
CALIFORNIA  UNITED STATES -> ('CALIFORNIA', 90)
CALIFORNIA  USA -> ('CALIFORNIA', 90)
CANADA -> ('NEVADA', 67)
CARSON CITY -> ('CONNECTICUT', 55)
CATHLAMET  WASHINGTON -> ('WASHINGTON', 90)
CHESHIRE HIGH SCHOOL  CHESHIRE  CONNECTICUT -> ('CONNECTICUT', 90)
CHESTER VIRGINIA -> ('VIRGINIA', 90)
CHESTERVILLE  ONTARIO  CANADA -> ('ARIZONA', 62)
CHICAGO -> ('OHIO', 55)
CINCINNATI OHIO -> ('OHIO', 90)
CINCINNATI  OHIO -> ('OHIO', 90)
COLORADO -> ('COLORADO', 100)
COLORADO -> ('COLORADO', 100)
COLUMBIA  SOUTH CAROLINA -> ('SOUTH CAROLINA', 90)
COLUMBUS  OHIO -> ('OHIO', 90)
CONNECTICUT -> ('CONNECTICUT', 100)
CONNECTICUT USA -> ('CONNECTICUT', 95)
CONVERSE -> ('NEW JERSEY', 56)
CONNECTICUT -> ('CONNECTICUT', 100)
CONNECTICUT  USA -> ('CONNECTICUT', 95)
DALLAS  TEXAS  USA -> ('TEXAS', 90)
DALTON HIGH  DALTON  GEORGIA -> ('GEORGIA', 90)
DANVILLE -> ('MAINE', 46)
DECATUR  ALABAMA -> ('ALABAMA', 90)
DELAWARE -> ('DELAWARE', 100)
DENVER  COLORADO -> ('COLORADO', 90)
DES MOINES  IOWA -> ('IOWA', 90)
DEXTER  MAINE USA -> ('MAINE', 90)
DILLSBURG PENNSYLVANIA -> ('PENNSYLVANIA', 90)
DOTHAN -> ('MONTANA', 62)
DUVAL -> ('GUAM', 44)
EDEN  NORTH CAROLINA -> ('NORTH CAROLINA', 95)
EL PASO  TEXAS -> ('TEXAS', 90)
FARMINGTON MICHIGAN -> ('MICHIGAN', 90)
FLORIDA -> ('FLORIDA', 100)
FLORIDA -> ('FLORIDA', 100)
FLORIDA  USA -> ('FLORIDA', 90)
FLORIDA  USA -> ('FLORIDA', 90)
FLUSHING  MICHIGAN -> ('MICHIGAN', 90)
FRAMINGHAM -> ('GUAM', 68)
FRANKLIN  NORTH CAROLINA -> ('NORTH CAROLINA', 90)
FT LAUDERDALE -> ('FLORIDA', 50)
GEORGIA -> ('GEORGIA', 100)
GEORGIA USA -> ('GEORGIA', 90)
GEORGIA US -> ('GEORGIA', 95)
GEORGIA -> ('GEORGIA', 100)
GEORGIA  US -> ('GEORGIA', 90)
GEORGIA  USA -> ('GEORGIA', 90)
GERMANY -> ('GEORGIA', 57)
GROSSE POINTE  MICHIGAN  UNITED STATES -> ('MICHIGAN', 90)
HAMPSHIRE HIGH  HAMPSHIRE  ILLINOIS -> ('ILLINOIS', 90)
HAWAII -> ('HAWAII', 100)
HENDERSON  NEVADA -> ('NEVADA', 90)
HENRICO  VIRGINIA  USA -> ('VIRGINIA', 90)
HAWAII -> ('HAWAII', 100)
HOFFMAN ESTATES  ILLINOIS -> ('ILLINOIS', 90)
HOUSTON TEXAS -> ('TEXAS', 90)
IDAHO -> ('IDAHO', 100)
IDAHO  USA -> ('IDAHO', 90)
ILLINOIS -> ('ILLINOIS', 100)
ILLINOIS -> ('ILLINOIS', 100)
ILLINOIS  USA -> ('ILLINOIS', 90)
INDIANA -> ('INDIANA', 100)
INDEPENDENCE  MISSOURI USA -> ('MISSOURI', 90)
INDIA -> ('INDIANA', 83)
INDIANA -> ('INDIANA', 100)
INDIANA  USA -> ('INDIANA', 90)
IOWA -> ('IOWA', 100)
IOWA  UNITED STATES -> ('IOWA', 90)
IOWA  UNITED STATES -> ('IOWA', 90)
KANSAS -> ('KANSAS', 100)
KENTUCKY -> ('KENTUCKY', 100)
KODIAK  ALASKA  USA -> ('ALASKA', 90)
KANSAS  USA -> ('KANSAS', 90)
KENTUCKY -> ('KENTUCKY', 100)
KENTUCKY  USA -> ('KENTUCKY', 90)
LOUISIANA -> ('LOUISIANA', 100)
LAKE BUTLER FLORIDA -> ('FLORIDA', 90)
LAKE MARY  FLORIDA -> ('FLORIDA', 90)
LAKELAND FLORIDA -> ('FLORIDA', 90)
LAKEWOOD  COLORADO -> ('COLORADO', 90)
LEXINGTON  SOUTH CAROLINA -> ('SOUTH CAROLINA', 90)
LITHONIA -> ('CALIFORNIA', 67)
LIVE OAK FLORIDA -> ('FLORIDA', 90)
LIVINGSTON  TEXAS -> ('TEXAS', 90)
LOS ANGELES  CALIFORNIA -> ('CALIFORNIA', 90)
LOUISIANA -> ('LOUISIANA', 100)
LOUISIANA  USA -> ('LOUISIANA', 90)
LOUISVILLE KENTUCKY -> ('KENTUCKY', 90)
LOUSIANA -> ('LOUISIANA', 94)
MASSACHUSETTS -> ('MASSACHUSETTS', 100)
MASSACHUSETTS USA -> ('MASSACHUSETTS', 95)
MASSACHUSETTS  USA -> ('MASSACHUSETTS', 95)
MAINE -> ('MAINE', 100)
MAINE  USA -> ('MAINE', 90)
MALAYSIA -> ('ALASKA', 71)
MALVERNE NEW YORK -> ('NEW YORK', 90)
MARYLAND -> ('MARYLAND', 100)
MARYLAND USA -> ('MARYLAND', 90)
MARYLAND  USA -> ('MARYLAND', 90)
MARYLAND USA -> ('MARYLAND', 90)
MASSACHUSETTES -> ('MASSACHUSETTS', 96)
MASSACHUSETTS -> ('MASSACHUSETTS', 100)
MARYLAND -> ('MARYLAND', 100)
MEADOWBROOK HIGHSCHOOL -> ('COLORADO', 53)
MICHIGAN -> ('MICHIGAN', 100)
MICHIGAN AND TENNESSEE -> ('MICHIGAN', 90)
MIAMI -> ('DISTRICT OF COLUMBIA', 60)
MICHIGAN -> ('MICHIGAN', 100)
MICHIGAN (US) -> ('MICHIGAN', 90)
MICHIGAN SENIOR YEAR INDIANA INDIANA THROUGH CORRESPONDENCE (STILL LIVED INDIANA MICHIGAN) -> ('INDIANA', 60)
MICHIGAN USA -> ('MICHIGAN', 90)
MICHIGAN  UNITED STATES -> ('MICHIGAN', 90)
MICHIGAN  US -> ('MICHIGAN', 90)
MICHIGAN  USA -> ('MICHIGAN', 90)
MILLINOCKET -> ('ILLINOIS', 63)
MINNESOTA -> ('MINNESOTA', 100)
MINNESOTA USA -> ('MINNESOTA', 95)
MISSISSIPPI -> ('MISSISSIPPI', 100)
MISSOURI -> ('MISSOURI', 100)
MISSOURI  UNITED STATES -> ('MISSOURI', 90)
MISSOURI  USA -> ('MISSOURI', 90)
MINNESOTA -> ('MINNESOTA', 100)
MISSOURI -> ('MISSOURI', 100)
MISSOURI USA -> ('MISSOURI', 90)
MODERN COLLEGE INDIA -> ('INDIANA', 75)
MONTANA -> ('MONTANA', 100)
MONTGOMERY COUNTY  PENNSYLVANIA -> ('PENNSYLVANIA', 90)
MULTIPLE LOCATIONS -> ('ARIZONA', 56)
MUSKEGON HEIGHTS  MICHIGAN -> ('MICHIGAN', 90)
MUSTANG  OKLAHOMA -> ('OKLAHOMA', 90)
NAUGATUCK WEST VIRGINIA -> ('VIRGINIA', 90)
NORTH CAROLINA -> ('NORTH CAROLINA', 100)
NORTH CAROLINA  USA -> ('NORTH CAROLINA', 95)
NDIA -> ('INDIANA', 90)
NEBRASKA -> ('NEBRASKA', 100)
NEVADA -> ('NEVADA', 100)
NEW  JERSEY USA -> ('NEW JERSEY', 86)
NEW  YORK -> ('NEW YORK', 95)
NEW CASTLE -> ('NEW HAMPSHIRE', 61)
NEW HAMPSHIRE -> ('NEW HAMPSHIRE', 100)
NEW JERSEY -> ('NEW JERSEY', 100)
NEW MEXICO -> ('NEW MEXICO', 100)
NEW YORK -> ('NEW YORK', 100)
NEW YORK CIRY -> ('NEW YORK', 90)
NEW YORK CITY -> ('NEW YORK', 90)
NEW YORK STATE -> ('NEW YORK', 90)
NEW YORK USA -> ('NEW YORK', 90)
NEW YORK  UNITED STATES -> ('NEW YORK', 90)
NEW YORK  US -> ('NEW YORK', 90)
NEW YORK  USA -> ('NEW YORK', 90)
NEW ZEALAND -> ('MARYLAND', 53)
NEW HAMPSHIRE -> ('NEW HAMPSHIRE', 100)
NEW JERSEY -> ('NEW JERSEY', 100)
NEW JERSEY  USA -> ('NEW JERSEY', 90)
NORFOLK  VIRGINIA -> ('VIRGINIA', 90)
NORTH CAROLINA -> ('NORTH CAROLINA', 100)
NORTH CAROLINA USA -> ('NORTH CAROLINA', 95)
NORTH CAROLINA  UNITED STATES -> ('NORTH CAROLINA', 90)
NORTH CAROLINA  USA -> ('NORTH CAROLINA', 95)
NORTH FLORIDA -> ('FLORIDA', 90)
NORWAY -> ('IOWA', 68)
NEW YORK -> ('NEW YORK', 100)
NEW YORK USA -> ('NEW YORK', 90)
NEW YORK  USA -> ('NEW YORK', 90)
NYC  NEW YORK -> ('NEW YORK', 90)
OCALA  FLORIDA -> ('FLORIDA', 90)
OHIO -> ('OHIO', 100)
OHIO -> ('OHIO', 100)
OHIO  USA -> ('OHIO', 90)
OKLAHOMA -> ('OKLAHOMA', 100)
ORANGE COUNTY  CALIFORNIA -> ('CALIFORNIA', 90)
ORANGE  CALIFORNIA -> ('CALIFORNIA', 90)
ORANGE  TEXAS -> ('TEXAS', 90)
OREGON -> ('OREGON', 100)
OREGON  USA -> ('OREGON', 90)
PENNSYLVANIA -> ('PENNSYLVANIA', 100)
PENNSYLVANIA  USA -> ('PENNSYLVANIA', 95)
PARADISE  CALIFORNIA -> ('CALIFORNIA', 90)
PARAMUS  NEW JERSEY -> ('NEW JERSEY', 90)
PEARL CITY  HAWAII -> ('HAWAII', 90)
PENNSYLVANIA -> ('PENNSYLVANIA', 100)
PENNSYLVANIA  UNITED STATES -> ('PENNSYLVANIA', 90)
PENNSYLVANIA  US -> ('PENNSYLVANIA', 95)
PENNSYLVANIA  USA -> ('PENNSYLVANIA', 95)
PENNSYVANIA -> ('PENNSYLVANIA', 96)
PHILADELPHIA  PENNSYLVANIA -> ('PENNSYLVANIA', 90)
PHILIPPINES -> ('MAINE', 54)
PITTSBURGH -> ('UTAH', 45)
PITTSBURGH  PENNSYLVANIA -> ('PENNSYLVANIA', 90)
PLANT CITY  FLORIDA -> ('FLORIDA', 90)
PORT HURON  MICHIGAN -> ('MICHIGAN', 90)
PUERTO RICO -> ('PUERTO RICO', 100)
QUARTZ HILL -> ('ARIZONA', 47)
QUEENS  NEW YORK -> ('NEW YORK', 90)
RHODE ISLAND -> ('RHODE ISLAND', 100)
RHODE ISLAND -> ('RHODE ISLAND', 100)
RICHMOND VIRGINIA -> ('VIRGINIA', 90)
RICHMOND  INDIANA -> ('INDIANA', 90)
ROCHESTER  NEW YORK -> ('NEW YORK', 90)
ROCK ISLAND  ILLINOIS -> ('ILLINOIS', 90)
ROGERS  ARKANSAS -> ('ARKANSAS', 90)
SAN DIEGO  CALIFORNIA  USA -> ('CALIFORNIA', 90)
SANTA CLARA -> ('ALASKA', 57)
SOUTH CAROLINA -> ('SOUTH CAROLINA', 100)
SOUTH CAROLINA   US -> ('SOUTH CAROLINA', 95)
SOUTH DAKOTA -> ('SOUTH DAKOTA', 100)
SENECA HIGH -> ('NEVADA', 51)
SHEBOYGAN  WISCONSIN -> ('WISCONSIN', 90)
SINGAPORE -> ('OREGON', 60)
SLOVAKIA -> ('PENNSYLVANIA', 68)
SOUTH CAROLINA -> ('SOUTH CAROLINA', 100)
SOUTH CAROLINA USA -> ('SOUTH CAROLINA', 95)
SOUTH DAKOTA -> ('SOUTH DAKOTA', 100)
SOUTHRIDGE HIGH SCHOOL -> ('UTAH', 68)
SPRING MILLS PENNSYLVANIA -> ('PENNSYLVANIA', 90)
SPRINGFIELD  MASSACHUSETTS -> ('MASSACHUSETTS', 90)
ST THOMAS  US VIRGIN ISLANDS -> ('VIRGIN ISLANDS', 90)
STPETE FLORIDA -> ('FLORIDA', 90)
TAMPA  FLORIDA -> ('FLORIDA', 90)
TENNESEE -> ('TENNESSEE', 94)
TENNESSEE -> ('TENNESSEE', 100)
TENNESSEE  USA -> ('TENNESSEE', 90)
TEXAS -> ('TEXAS', 100)
TEXAS  USA -> ('TEXAS', 90)
TEXAS USA -> ('TEXAS', 90)
TENNESSEE -> ('TENNESSEE', 100)
TROY  MONTANA -> ('MONTANA', 90)
TEXAS -> ('TEXAS', 100)
UNCASVILLE  CONNECTICUT -> ('CONNECTICUT', 90)
UNITED STATES -> ('TEXAS', 54)
UNITED STATES OF - -> ('TEXAS', 54)
UNITED STATES  ILLINOIS -> ('ILLINOIS', 90)
UNITED STATES  MINNESOTA -> ('MINNESOTA', 90)
US -> ('MASSACHUSETTS', 90)
US  NEW YORK -> ('NEW YORK', 90)
USA -> ('ALASKA', 60)
USA FLA -> ('ALASKA', 59)
USA  NEW JERSEY -> ('NEW JERSEY', 90)
UTAH -> ('UTAH', 100)
UTAH  UNITED STATES OF - -> ('UTAH', 90)
UTAH  USA -> ('UTAH', 90)
VIRGINIA -> ('VIRGINIA', 100)
VIRGINIA  USA -> ('VIRGINIA', 90)
VERMONT -> ('VERMONT', 100)
VERMONT USA -> ('VERMONT', 90)
VERMONT  UNITED STATES -> ('VERMONT', 90)
VIRGINIA -> ('VIRGINIA', 100)
VIRGINIA  USA -> ('VIRGINIA', 90)
WASHINGTON -> ('WASHINGTON', 100)
WASHINGTON STATE -> ('WASHINGTON', 90)
WALLA WALLA  WASHINGTON -> ('WASHINGTON', 90)
WASHINGTON -> ('WASHINGTON', 100)
WASHINGTON DISTRICT OF COLUMBIA -> ('DISTRICT OF COLUMBIA', 90)
WASHINGTON STATE -> ('WASHINGTON', 90)
WASHINGTON STATE  USA -> ('WASHINGTON', 90)
WATERFORD  MICHIGAN -> ('MICHIGAN', 90)
WAYNESBORO VIRGINIA -> ('VIRGINIA', 90)
WEST VIRGINIA -> ('WEST VIRGINIA', 100)
WESTFIELD NEW JERSEY -> ('NEW JERSEY', 90)
WHEELING  WEST V IRGINIA UNITED STATES -> ('WEST VIRGINIA', 86)
WHITWELL  TENNESSEE -> ('TENNESSEE', 90)
WISCONSIN -> ('WISCONSIN', 100)
WISCONSIN  USA -> ('WISCONSIN', 90)
WISCONSIN -> ('WISCONSIN', 100)
WORTHINGTON -> ('WASHINGTON', 76)
WEST VIRGINIA -> ('WEST VIRGINIA', 100)
YAKIMA WASHINGTON -> ('WASHINGTON', 90)
YOUNGSTOWN OHIO -> ('OHIO', 90)
''')

st.write('''
The code below shows the good matches for values greater than or equal to 86.
''')
st.code('''
for string_value in new_states_list:
    best_match = process.extractOne(string_value, us_state_name_list)
    if best_match[1] >= 86:
        print(f"{string_value} -> {best_match}")
''')
st.code('''
ALABAMA -> ('ALABAMA', 100)
ALABAMA -> ('ALABAMA', 100)
ALABAMA  USA -> ('ALABAMA', 90)
ALASKA -> ('ALASKA', 100)
ARKANSAS  USA -> ('ARKANSAS', 90)
ARIZONA -> ('ARIZONA', 100)
ARIZONA  USA -> ('ARIZONA', 90)
ARKANSAS -> ('ARKANSAS', 100)
ARKANSAS  USA -> ('ARKANSAS', 90)
ATLANTA  GEORGIA -> ('GEORGIA', 90)
ARIZONA -> ('ARIZONA', 100)
BOSTON MASSACHUSETTS -> ('MASSACHUSETTS', 90)
BRISTOL  CONNECTICUT -> ('CONNECTICUT', 90)
CALIFORNIA -> ('CALIFORNIA', 100)
CALIFORNIA  USA -> ('CALIFORNIA', 90)
CABOT  ARKANSAS -> ('ARKANSAS', 90)
CAFLIFORNIA -> ('CALIFORNIA', 95)
CALIFORNI -> ('CALIFORNIA', 95)
CALIFORNIA -> ('CALIFORNIA', 100)
CALIFORNIA USA -> ('CALIFORNIA', 95)
CALIFORNIA  UNITED STATES -> ('CALIFORNIA', 90)
CALIFORNIA  USA -> ('CALIFORNIA', 90)
CATHLAMET  WASHINGTON -> ('WASHINGTON', 90)
CHESHIRE HIGH SCHOOL  CHESHIRE  CONNECTICUT -> ('CONNECTICUT', 90)
CHESTER VIRGINIA -> ('VIRGINIA', 90)
CINCINNATI OHIO -> ('OHIO', 90)
CINCINNATI  OHIO -> ('OHIO', 90)
COLORADO -> ('COLORADO', 100)
COLORADO -> ('COLORADO', 100)
COLUMBIA  SOUTH CAROLINA -> ('SOUTH CAROLINA', 90)
COLUMBUS  OHIO -> ('OHIO', 90)
CONNECTICUT -> ('CONNECTICUT', 100)
CONNECTICUT USA -> ('CONNECTICUT', 95)
CONNECTICUT -> ('CONNECTICUT', 100)
CONNECTICUT  USA -> ('CONNECTICUT', 95)
DALLAS  TEXAS  USA -> ('TEXAS', 90)
DALTON HIGH  DALTON  GEORGIA -> ('GEORGIA', 90)
DECATUR  ALABAMA -> ('ALABAMA', 90)
DELAWARE -> ('DELAWARE', 100)
DENVER  COLORADO -> ('COLORADO', 90)
DES MOINES  IOWA -> ('IOWA', 90)
DEXTER  MAINE USA -> ('MAINE', 90)
DILLSBURG PENNSYLVANIA -> ('PENNSYLVANIA', 90)
EDEN  NORTH CAROLINA -> ('NORTH CAROLINA', 95)
EL PASO  TEXAS -> ('TEXAS', 90)
FARMINGTON MICHIGAN -> ('MICHIGAN', 90)
FLORIDA -> ('FLORIDA', 100)
FLORIDA -> ('FLORIDA', 100)
FLORIDA  USA -> ('FLORIDA', 90)
FLORIDA  USA -> ('FLORIDA', 90)
FLUSHING  MICHIGAN -> ('MICHIGAN', 90)
FRANKLIN  NORTH CAROLINA -> ('NORTH CAROLINA', 90)
GEORGIA -> ('GEORGIA', 100)
GEORGIA USA -> ('GEORGIA', 90)
GEORGIA US -> ('GEORGIA', 95)
GEORGIA -> ('GEORGIA', 100)
GEORGIA  US -> ('GEORGIA', 90)
GEORGIA  USA -> ('GEORGIA', 90)
GROSSE POINTE  MICHIGAN  UNITED STATES -> ('MICHIGAN', 90)
HAMPSHIRE HIGH  HAMPSHIRE  ILLINOIS -> ('ILLINOIS', 90)
HAWAII -> ('HAWAII', 100)
HENDERSON  NEVADA -> ('NEVADA', 90)
HENRICO  VIRGINIA  USA -> ('VIRGINIA', 90)
HAWAII -> ('HAWAII', 100)
HOFFMAN ESTATES  ILLINOIS -> ('ILLINOIS', 90)
HOUSTON TEXAS -> ('TEXAS', 90)
IDAHO -> ('IDAHO', 100)
IDAHO  USA -> ('IDAHO', 90)
ILLINOIS -> ('ILLINOIS', 100)
ILLINOIS -> ('ILLINOIS', 100)
ILLINOIS  USA -> ('ILLINOIS', 90)
INDIANA -> ('INDIANA', 100)
INDEPENDENCE  MISSOURI USA -> ('MISSOURI', 90)
INDIANA -> ('INDIANA', 100)
INDIANA  USA -> ('INDIANA', 90)
IOWA -> ('IOWA', 100)
IOWA  UNITED STATES -> ('IOWA', 90)
IOWA  UNITED STATES -> ('IOWA', 90)
KANSAS -> ('KANSAS', 100)
KENTUCKY -> ('KENTUCKY', 100)
KODIAK  ALASKA  USA -> ('ALASKA', 90)
KANSAS  USA -> ('KANSAS', 90)
KENTUCKY -> ('KENTUCKY', 100)
KENTUCKY  USA -> ('KENTUCKY', 90)
LOUISIANA -> ('LOUISIANA', 100)
LAKE BUTLER FLORIDA -> ('FLORIDA', 90)
LAKE MARY  FLORIDA -> ('FLORIDA', 90)
LAKELAND FLORIDA -> ('FLORIDA', 90)
LAKEWOOD  COLORADO -> ('COLORADO', 90)
LEXINGTON  SOUTH CAROLINA -> ('SOUTH CAROLINA', 90)
LIVE OAK FLORIDA -> ('FLORIDA', 90)
LIVINGSTON  TEXAS -> ('TEXAS', 90)
LOS ANGELES  CALIFORNIA -> ('CALIFORNIA', 90)
LOUISIANA -> ('LOUISIANA', 100)
LOUISIANA  USA -> ('LOUISIANA', 90)
LOUISVILLE KENTUCKY -> ('KENTUCKY', 90)
LOUSIANA -> ('LOUISIANA', 94)
MASSACHUSETTS -> ('MASSACHUSETTS', 100)
MASSACHUSETTS USA -> ('MASSACHUSETTS', 95)
MASSACHUSETTS  USA -> ('MASSACHUSETTS', 95)
MAINE -> ('MAINE', 100)
MAINE  USA -> ('MAINE', 90)
MALVERNE NEW YORK -> ('NEW YORK', 90)
MARYLAND -> ('MARYLAND', 100)
MARYLAND USA -> ('MARYLAND', 90)
MARYLAND  USA -> ('MARYLAND', 90)
MARYLAND USA -> ('MARYLAND', 90)
MASSACHUSETTES -> ('MASSACHUSETTS', 96)
MASSACHUSETTS -> ('MASSACHUSETTS', 100)
MARYLAND -> ('MARYLAND', 100)
MICHIGAN -> ('MICHIGAN', 100)
MICHIGAN AND TENNESSEE -> ('MICHIGAN', 90)
MICHIGAN -> ('MICHIGAN', 100)
MICHIGAN (US) -> ('MICHIGAN', 90)
MICHIGAN USA -> ('MICHIGAN', 90)
MICHIGAN  UNITED STATES -> ('MICHIGAN', 90)
MICHIGAN  US -> ('MICHIGAN', 90)
MICHIGAN  USA -> ('MICHIGAN', 90)
MINNESOTA -> ('MINNESOTA', 100)
MINNESOTA USA -> ('MINNESOTA', 95)
MISSISSIPPI -> ('MISSISSIPPI', 100)
MISSOURI -> ('MISSOURI', 100)
MISSOURI  UNITED STATES -> ('MISSOURI', 90)
MISSOURI  USA -> ('MISSOURI', 90)
MINNESOTA -> ('MINNESOTA', 100)
MISSOURI -> ('MISSOURI', 100)
MISSOURI USA -> ('MISSOURI', 90)
MONTANA -> ('MONTANA', 100)
MONTGOMERY COUNTY  PENNSYLVANIA -> ('PENNSYLVANIA', 90)
MUSKEGON HEIGHTS  MICHIGAN -> ('MICHIGAN', 90)
MUSTANG  OKLAHOMA -> ('OKLAHOMA', 90)
NAUGATUCK WEST VIRGINIA -> ('VIRGINIA', 90)
NORTH CAROLINA -> ('NORTH CAROLINA', 100)
NORTH CAROLINA  USA -> ('NORTH CAROLINA', 95)
NDIA -> ('INDIANA', 90)
NEBRASKA -> ('NEBRASKA', 100)
NEVADA -> ('NEVADA', 100)
NEW  JERSEY USA -> ('NEW JERSEY', 86)
NEW  YORK -> ('NEW YORK', 95)
NEW HAMPSHIRE -> ('NEW HAMPSHIRE', 100)
NEW JERSEY -> ('NEW JERSEY', 100)
NEW MEXICO -> ('NEW MEXICO', 100)
NEW YORK -> ('NEW YORK', 100)
NEW YORK CIRY -> ('NEW YORK', 90)
NEW YORK CITY -> ('NEW YORK', 90)
NEW YORK STATE -> ('NEW YORK', 90)
NEW YORK USA -> ('NEW YORK', 90)
NEW YORK  UNITED STATES -> ('NEW YORK', 90)
NEW YORK  US -> ('NEW YORK', 90)
NEW YORK  USA -> ('NEW YORK', 90)
NEW HAMPSHIRE -> ('NEW HAMPSHIRE', 100)
NEW JERSEY -> ('NEW JERSEY', 100)
NEW JERSEY  USA -> ('NEW JERSEY', 90)
NORFOLK  VIRGINIA -> ('VIRGINIA', 90)
NORTH CAROLINA -> ('NORTH CAROLINA', 100)
NORTH CAROLINA USA -> ('NORTH CAROLINA', 95)
NORTH CAROLINA  UNITED STATES -> ('NORTH CAROLINA', 90)
NORTH CAROLINA  USA -> ('NORTH CAROLINA', 95)
NORTH FLORIDA -> ('FLORIDA', 90)
NEW YORK -> ('NEW YORK', 100)
NEW YORK USA -> ('NEW YORK', 90)
NEW YORK  USA -> ('NEW YORK', 90)
NYC  NEW YORK -> ('NEW YORK', 90)
OCALA  FLORIDA -> ('FLORIDA', 90)
OHIO -> ('OHIO', 100)
OHIO -> ('OHIO', 100)
OHIO  USA -> ('OHIO', 90)
OKLAHOMA -> ('OKLAHOMA', 100)
ORANGE COUNTY  CALIFORNIA -> ('CALIFORNIA', 90)
ORANGE  CALIFORNIA -> ('CALIFORNIA', 90)
ORANGE  TEXAS -> ('TEXAS', 90)
OREGON -> ('OREGON', 100)
OREGON  USA -> ('OREGON', 90)
PENNSYLVANIA -> ('PENNSYLVANIA', 100)
PENNSYLVANIA  USA -> ('PENNSYLVANIA', 95)
PARADISE  CALIFORNIA -> ('CALIFORNIA', 90)
PARAMUS  NEW JERSEY -> ('NEW JERSEY', 90)
PEARL CITY  HAWAII -> ('HAWAII', 90)
PENNSYLVANIA -> ('PENNSYLVANIA', 100)
PENNSYLVANIA  UNITED STATES -> ('PENNSYLVANIA', 90)
PENNSYLVANIA  US -> ('PENNSYLVANIA', 95)
PENNSYLVANIA  USA -> ('PENNSYLVANIA', 95)
PENNSYVANIA -> ('PENNSYLVANIA', 96)
PHILADELPHIA  PENNSYLVANIA -> ('PENNSYLVANIA', 90)
PITTSBURGH  PENNSYLVANIA -> ('PENNSYLVANIA', 90)
PLANT CITY  FLORIDA -> ('FLORIDA', 90)
PORT HURON  MICHIGAN -> ('MICHIGAN', 90)
PUERTO RICO -> ('PUERTO RICO', 100)
QUEENS  NEW YORK -> ('NEW YORK', 90)
RHODE ISLAND -> ('RHODE ISLAND', 100)
RHODE ISLAND -> ('RHODE ISLAND', 100)
RICHMOND VIRGINIA -> ('VIRGINIA', 90)
RICHMOND  INDIANA -> ('INDIANA', 90)
ROCHESTER  NEW YORK -> ('NEW YORK', 90)
ROCK ISLAND  ILLINOIS -> ('ILLINOIS', 90)
ROGERS  ARKANSAS -> ('ARKANSAS', 90)
SAN DIEGO  CALIFORNIA  USA -> ('CALIFORNIA', 90)
SOUTH CAROLINA -> ('SOUTH CAROLINA', 100)
SOUTH CAROLINA   US -> ('SOUTH CAROLINA', 95)
SOUTH DAKOTA -> ('SOUTH DAKOTA', 100)
SHEBOYGAN  WISCONSIN -> ('WISCONSIN', 90)
SOUTH CAROLINA -> ('SOUTH CAROLINA', 100)
SOUTH CAROLINA USA -> ('SOUTH CAROLINA', 95)
SOUTH DAKOTA -> ('SOUTH DAKOTA', 100)
SPRING MILLS PENNSYLVANIA -> ('PENNSYLVANIA', 90)
SPRINGFIELD  MASSACHUSETTS -> ('MASSACHUSETTS', 90)
ST THOMAS  US VIRGIN ISLANDS -> ('VIRGIN ISLANDS', 90)
STPETE FLORIDA -> ('FLORIDA', 90)
TAMPA  FLORIDA -> ('FLORIDA', 90)
TENNESEE -> ('TENNESSEE', 94)
TENNESSEE -> ('TENNESSEE', 100)
TENNESSEE  USA -> ('TENNESSEE', 90)
TEXAS -> ('TEXAS', 100)
TEXAS  USA -> ('TEXAS', 90)
TEXAS USA -> ('TEXAS', 90)
TENNESSEE -> ('TENNESSEE', 100)
TROY  MONTANA -> ('MONTANA', 90)
TEXAS -> ('TEXAS', 100)
UNCASVILLE  CONNECTICUT -> ('CONNECTICUT', 90)
UNITED STATES  ILLINOIS -> ('ILLINOIS', 90)
UNITED STATES  MINNESOTA -> ('MINNESOTA', 90)
US -> ('MASSACHUSETTS', 90)
US  NEW YORK -> ('NEW YORK', 90)
USA  NEW JERSEY -> ('NEW JERSEY', 90)
UTAH -> ('UTAH', 100)
UTAH  UNITED STATES OF - -> ('UTAH', 90)
UTAH  USA -> ('UTAH', 90)
VIRGINIA -> ('VIRGINIA', 100)
VIRGINIA  USA -> ('VIRGINIA', 90)
VERMONT -> ('VERMONT', 100)
VERMONT USA -> ('VERMONT', 90)
VERMONT  UNITED STATES -> ('VERMONT', 90)
VIRGINIA -> ('VIRGINIA', 100)
VIRGINIA  USA -> ('VIRGINIA', 90)
WASHINGTON -> ('WASHINGTON', 100)
WASHINGTON STATE -> ('WASHINGTON', 90)
WALLA WALLA  WASHINGTON -> ('WASHINGTON', 90)
WASHINGTON -> ('WASHINGTON', 100)
WASHINGTON DISTRICT OF COLUMBIA -> ('DISTRICT OF COLUMBIA', 90)
WASHINGTON STATE -> ('WASHINGTON', 90)
WASHINGTON STATE  USA -> ('WASHINGTON', 90)
WATERFORD  MICHIGAN -> ('MICHIGAN', 90)
WAYNESBORO VIRGINIA -> ('VIRGINIA', 90)
WEST VIRGINIA -> ('WEST VIRGINIA', 100)
WESTFIELD NEW JERSEY -> ('NEW JERSEY', 90)
WHEELING  WEST V IRGINIA UNITED STATES -> ('WEST VIRGINIA', 86)
WHITWELL  TENNESSEE -> ('TENNESSEE', 90)
WISCONSIN -> ('WISCONSIN', 100)
WISCONSIN  USA -> ('WISCONSIN', 90)
WISCONSIN -> ('WISCONSIN', 100)
WEST VIRGINIA -> ('WEST VIRGINIA', 100)
YAKIMA WASHINGTON -> ('WASHINGTON', 90)
YOUNGSTOWN OHIO -> ('OHIO', 90)
''')

st.write('''
The code below shows the poor matches, corresponding to values less than 86. All these string inputs will be assigned later to the `NA` abbrviation.
''')
st.code('''
for string_value in new_states_list:
    best_match = process.extractOne(string_value, us_state_name_list)
    if best_match[1] < 86:
        print(f"{string_value} -> {best_match}")
''')
st.code('''
- -> ('ALASKA', 0)
- -> ('ALASKA', 0)
ARMY BASE PISA ITALY  LIVORNO UNIT HIGHSCHOOL -> ('LOUISIANA', 48)
ARUBA -> ('ARKANSAS', 54)
BUENOS AIRES  ARGENTINA -> ('VIRGINIA', 56)
CANADA -> ('NEVADA', 67)
CARSON CITY -> ('CONNECTICUT', 55)
CHESTERVILLE  ONTARIO  CANADA -> ('ARIZONA', 62)
CHICAGO -> ('OHIO', 55)
CONVERSE -> ('NEW JERSEY', 56)
DANVILLE -> ('MAINE', 46)
DOTHAN -> ('MONTANA', 62)
DUVAL -> ('GUAM', 44)
FRAMINGHAM -> ('GUAM', 68)
FT LAUDERDALE -> ('FLORIDA', 50)
GERMANY -> ('GEORGIA', 57)
INDIA -> ('INDIANA', 83)
LITHONIA -> ('CALIFORNIA', 67)
MALAYSIA -> ('ALASKA', 71)
MEADOWBROOK HIGHSCHOOL -> ('COLORADO', 53)
MIAMI -> ('DISTRICT OF COLUMBIA', 60)
MICHIGAN SENIOR YEAR INDIANA INDIANA THROUGH CORRESPONDENCE (STILL LIVED INDIANA MICHIGAN) -> ('INDIANA', 60)
MILLINOCKET -> ('ILLINOIS', 63)
MODERN COLLEGE INDIA -> ('INDIANA', 75)
MULTIPLE LOCATIONS -> ('ARIZONA', 56)
NEW CASTLE -> ('NEW HAMPSHIRE', 61)
NEW ZEALAND -> ('MARYLAND', 53)
NORWAY -> ('IOWA', 68)
PHILIPPINES -> ('MAINE', 54)
PITTSBURGH -> ('UTAH', 45)
QUARTZ HILL -> ('ARIZONA', 47)
SANTA CLARA -> ('ALASKA', 57)
SENECA HIGH -> ('NEVADA', 51)
SINGAPORE -> ('OREGON', 60)
SLOVAKIA -> ('PENNSYLVANIA', 68)
SOUTHRIDGE HIGH SCHOOL -> ('UTAH', 68)
UNITED STATES -> ('TEXAS', 54)
UNITED STATES OF - -> ('TEXAS', 54)
USA -> ('ALASKA', 60)
USA FLA -> ('ALASKA', 59)
WORTHINGTON -> ('WASHINGTON', 76)
''')

st.write('''
Here\'s the complete code when we put all of this together. The revised first 5 rows in the data frame are shown below the code.
Note the new values in the `state` column.
''')
st.code('''
uppercase_us_state_to_abbrev = dict(zip(us_state_name_list, us_state_abbrev_list))

def normalize_state(state_string):
    temp = state_string.replace(".", "")
    temp = temp.upper()

    import re
    temp_list = temp.replace("USVI", "US VI")
    temp_list = re.split("[/, ]", temp_list)
    new_list = []
    for word in temp_list:
        if word in us_state_abbrev_list:
            new_list.append(str.upper(abbrev_to_us_state[word]))
        elif word == 'AMERICA':
            new_list.append('-')
        else:
            new_list.append(word)
    temp = str.strip(' '.join(new_list))

    from fuzzywuzzy import process
    best_match = process.extractOne(temp, us_state_name_list)
    if best_match[1] >= 86:
        return uppercase_us_state_to_abbrev[best_match[0]]
    else:
        return 'NA'

df['state'] = df['state'].apply(normalize_state)
''')

uppercase_us_state_to_abbrev = dict(zip(us_state_name_list, us_state_abbrev_list))

def normalize_state(state_string):
    temp = state_string.replace(".", "")
    temp = temp.upper()

    import re
    temp_list = temp.replace("USVI", "US VI")
    temp_list = re.split("[/, ]", temp_list)
    new_list = []
    for word in temp_list:
        if word in us_state_abbrev_list:
            new_list.append(str.upper(abbrev_to_us_state[word]))
        elif word == 'AMERICA':
            new_list.append('-')
        else:
            new_list.append(word)
    temp = str.strip(' '.join(new_list))

    from fuzzywuzzy import process
    best_match = process.extractOne(temp, us_state_name_list)
    if best_match[1] >= 86:
        return uppercase_us_state_to_abbrev[best_match[0]]
    else:
        return 'NA'

df['state'] = df['state'].apply(normalize_state)

st.table(data=df[0:5])


st.subheader('Part 3. Exploratory data analysis')
st.write('Let\'s explore the data some more. Here are the tasks in this part.')

st.write('''
* Look at the outliers in each column, using histograms
* Look at the outliers in each column, using boxplots
* For each column, use the kernel density estimate (kde) to plotting the shape of the distribution
* Use the `.corr()` function to find correlations using pandas. Then visualize the correlation matrix using a heatmap in seaborn
* Visualize the same correlation matrix but this time include the correlation value in the graph
* Use geopandas to display info about the data frame in the context of a US map.
* Use geopandas to determine if there are variations bewteen the five geographical US regions: West, Southwest, Southeast, Midwest, Northeast.
''')



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
