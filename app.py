import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(
                   page_title='Simple Prediction App',
                   layout='wide',
                   initial_sidebar_state='expanded'
                  )

# Title of the App
st.write('Simple ML Prediction App')

# Load Dataset
# df = pd.read_csv(r'C:\Users\shanm\OneDrive\Desktop\E2E\2_streamlitapp\iris.csv')
df = pd.read_csv('https://raw.githubusercontent.com/soundaryassj/2_streamlitapp/master/Iris.csv')
df.drop('Id',axis = 1, inplace=True)
st.write(df)

# Input Widgets
st.sidebar.subheader('Input Features')
sepal_length = st.sidebar.slider('Sepal Length',4.3,7.9,5.8)
sepal_width  = st.sidebar.slider('Sepal Width',2.0,4.4,3.1)
petal_length = st.sidebar.slider('Petal Width',1.0,6.9,3.8)
petal_width  = st.sidebar.slider('Petal Width',0.1,2.5,1.2)

# Separate X and y
X = df.drop('Species',axis =1)
y = df.Species

# Data Splitting
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Model Building
model = RandomForestClassifier(max_depth=2,
                               max_features=4,
                               n_estimators=200,
                               random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict([[sepal_length,
                         sepal_width,
                         petal_length,
                         petal_width]])

# Print EDA
st.subheader('Brief EDA')
st.write('The Data is grouped by the class and the variable mean is computed for each class.')
groupby_species_mean = df.groupby('Species').mean()
st.write(groupby_species_mean)
st.line_chart(groupby_species_mean.T)

# Print the Input Features
input_features = pd.DataFrame([[sepal_length,sepal_width,
                                petal_length,petal_width]],
                                columns=['sepal_length','sepal_width',
                                          'petal_length','petal_width'])
st.write(input_features)

# Print prediction output
st.subheader('Output')
st.metric('Predicted Class',y_pred[0],'')
#st.write()







           

