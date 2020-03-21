from src.features.build_features import handle_zeros, resampled_teams
from src.visualization.visualize import plot_divergence
from pathlib import Path
import pandas as pd
import streamlit as st


FILE_NAME = 'cycle3.csv'
PATH = Path('../data/processed')
processed = pd.read_csv(PATH / FILE_NAME)
processed = handle_zeros(processed)

st.title("Resampled Score By CFF")
question = st.sidebar.selectbox(
    "Counterfactual Forecast",
    sorted(processed.Question.unique()))
# for question in processed.Question.unique():
resampling_period = st.sidebar.radio(
    "Resampling Period",
    ['H (hourly)', 'D (daily)', 'B (business daily)'])
resampling_period = resampling_period.split(' ')[0]
scoring_method = st.sidebar.radio(
    "Scoring Method",
    ['FairSkill', 'SWRPS'])

fig = plot_divergence(processed, scoring_method, resampling_period, question)
st.plotly_chart(fig)

# show_data = st.checkbox('Show raw data')
# if show_data:
    # st.subheader('Kiwis')
    # kiwi
    # st.subheader('Mangoes')
    # mango
