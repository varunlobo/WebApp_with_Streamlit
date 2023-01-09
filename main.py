
import streamlit as st
import pandas as pd

header = st.container()
dataset = st.container()
features = st.container()
modeltraining = st.container()


with header:
    
    st.title("Hello World, Welcome to my page")
    st.text("This is a trial page to experiment with streamlit")
    

with dataset:
    st.header("This is the section for dataset")
    df=pd.read_csv("employee_data.csv")
    st.subheader("A view of our dataset")
    st.write(df.head(5))
    st.bar_chart(df['Age'].value_counts()) 
    
with features:
    st.header("This is the features section")
    st.markdown("This is a markdown")
    
with modeltraining:
    st.header("This is the model training section")
    st.text("Here you get to choose the hyperparameters of the model")
    
    sel_col, display_col = st.columns(2)
    
    sel_col.header('Hello')
    
    display_col.header('Hello')
    
    
    