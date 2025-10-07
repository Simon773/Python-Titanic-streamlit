import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_csv("data/train.csv")

st.title("Pyhton Project with Streamlit")

Onglet1, Onglet2 = st.tabs(["Data viz", "Gaussian"])

with Onglet1:
    st.title("Data Visualization Titanic")

    fig = fig = px.bar(
        df["Survived"].value_counts().reset_index(),
        x="Survived",
        y="count",
        title="Répartition des survivants  ",
    )
    st.plotly_chart(fig)

    df_sex = df.groupby(["Sex", "Survived"]).size().reset_index(name="count")

    fig2 = px.bar(
        df_sex,
        "Sex",
        "count",
        barmode="group",
        color="Survived",
        title="Survivants par sexe",
    )
    st.plotly_chart(fig2)
    #

    col = df.select_dtypes(include="number").columns.tolist()
    col.remove("PassengerId")
    selectbox = st.selectbox(
        "Select feature",
        col,
        key="selectbox",
    )
    fig3 = px.histogram(df, x=selectbox, title=f"Distribution de {selectbox}")
    st.plotly_chart(fig3)

with Onglet2:
    st.title("Multi Gaussian Distribution")
    col1, col2 = st.columns(2)
    with col1:
        mean1 = st.number_input("Mean 1", value=0.0, step=0.1, key="mean1")
        std1 = st.number_input("Standard Deviation 1", value=1.0, step=0.1, key="std1")
        n1 = st.number_input("Number of points 1", value=1000, step=100, key="n1")
    with col2:
        mean2 = st.number_input("Mean 2", value=5.0, step=0.1, key="mean2")
        std2 = st.number_input("Standard Deviation 2", value=1.0, step=0.1, key="std2")
        n2 = st.number_input("Number of points 2", value=1000, step=100, key="n2")
    x1 = np.random.normal(mean1, std1, n1)
    x2 = np.random.normal(mean2, std2, n2)
    x = np.concatenate([x1, x2])
    fig4 = px.histogram(x, nbins=50, title="Gaussian Mixture")
    st.plotly_chart(fig4)
