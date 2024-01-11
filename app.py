import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
svm = pickle.load(open('SVC.pkl','rb'))

#load dataset
data = pd.read_csv('heart_dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Heart')

html_layout1 = """
<br>
<div style="background-color:black ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Heart Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['KNN','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Pasien Heart</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

#if st.checkbox('EDa'):
#    pr =ProfileReport(data,explorative=True)
#    st.header('**Input Dataframe**')
#    st.write(data)
#    st.write('---')
#    st.header('**Profiling Report**')
#    st_profile_report(pr)

#train test split
X_new = data[['sex','cp','fbs','restecg','exng','oldpeak','slp','caa','thall']]
y_new = data['output']
X = data.drop('output',axis=1)
y = data['output']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
X_train, X_test,y_train,y_test = train_test_split(X_new,y_new,test_size=0.20,random_state=42)
svm.fit(X_train, y_train)

pickle.dump(svm, open('SVC_updated.pkl', 'wb'))

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    sex = st.sidebar.slider('sex',0,20,1)
    cp = st.sidebar.slider('cp',0,200,108)
    fbs = st.sidebar.slider('fbs',0,140,40)
    restecg = st.sidebar.slider('restecg',0,100,25)
    exng = st.sidebar.slider('exng',0,1000,120)
    oldpeak = st.sidebar.slider('oldpeak',0,80,25)
    slp = st.sidebar.slider('slp', 0.05,2.5,0.45)
    caa = st.sidebar.slider('caa',21,100,24)
    thall = st.sidebar.slider('thall',21,100,24)
    
    user_report_data = {
        'sex':sex,
        'cp':cp,
        'fbs':fbs,
        'restecg':restecg,
        'exng':exng,
        'oldpeak':oldpeak,
        'slp':slp,
        'caa':caa,
        'thall':thall,
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasien
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = svm.predict(user_data)
svc_score = accuracy_score(y_test,svm.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena diabetes'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')