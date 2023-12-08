from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
from PIL import Image
import streamlit as st
import pandas as pd

from ml import preprocess, preprocess_text, preprocess_text_nn, predict_text

# from streamlit_dimensions import st_dimensions
from streamlit_pills import pills

# from streamlit_profiler import Profiler

# profiler = Profiler()

st.set_page_config("Sentiment Analysis", "🎪", layout="wide")
NUM_COLS = 10

EXCLUDE = [
    "streamlit",
    "streamlit-nightly",
    "repl-streamlit",
    "streamlit-with-ssl",
    "streamlit-fesion",
    "streamlit-aggrid-pro",
    "st-dbscan",
    "st-kickoff",
    "st-undetected-chromedriver",
    "st-package-reviewer",
    "streamlit-webcam-example",
    "st-pyv8",
    "streamlit-extras-arnaudmiribel",
    "st-schema-python",
    "st-optics",
    "st-spin",
    "st-dataprovider",
    "st-microservice",
    "st_nester",
    "st-jsme",
    "st-parsetree",
    "st-git-hooks",
    "st-schema",
    "st-distributions",
    "st-common-data",
    "awesome-streamlit",
    "awesome-streamlit-master",
    "extra-streamlit-components-SEM",
    "barfi",
    "streamlit-plotly-events-retro",
    "pollination-streamlit-io",
    "pollination-streamlit-viewer",
    "st-clustering",
    "streamlit-text-rating-component",
    "custom-streamlit",
    "hf-streamlit",
]

CATEGORY_NAMES = {
    # Putting this first so people don't miss it. Plus I think's it's one of the most
    # important ones.
    "widgets": "General widgets",  # 35
    # Visualizations of different data types.
    "charts": "Charts",  # 16
    "image": "Images",  # 10
    "video": "Video",  # 6
    "text": "Text",  # 12
    "maps": "Maps & geospatial",  # 7
    "dataframe": "Dataframes & tables",  # 6
    "science": "Molecules & genes",  # 3
    "graph": "Graphs",  # 7
    "3d": "3D",  # 1
    "code": "Code & editors",  # 4
    # More general elements in the app.
    "navigation": "Page navigation",  # 12
    "authentication": "Authentication",  # 5
    "style": "Style & layout",  # 3
    # More backend-y/dev stuff.
    # TODO: Should probably split this up, "Developer tools" contains a lot of stuff.
    "development": "Developer tools",  # 22
    "app-builder": "App builders",  # 3
    # General purpose categories.
    "integrations": "Integrations with other tools",  # 14
    "collection": "Collections of components",  # 4
}




def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


st.write(
    '<style>button[title="View fullscreen"], h4 a {display: none !important} [data-testid="stImage"] img {border: 1px solid #D6D6D9; border-radius: 3px; height: 200px; object-fit: cover; width: 100%} .block-container img:hover {}</style>',
    unsafe_allow_html=True,
)


icon("✈️")
"""
# Airline Sentiment Analysis

[![](https://img.shields.io/badge/github-pranavajk1%2Fcse587--project--phase3-blue.svg?style=for-the-badge&logo=github)](https://github.com/pranavajk1/cse587-project-phase3) &nbsp;
"""


description_text = """
The objective of the problem is to identify the sentiment i.e., classify an airline review into positive, negative or neutral based on keywords in the sentence.
"""
description = st.empty()
description.write(description_text.format("all"))
col1, col2, col3 = st.columns([3, 2, 1])
airline_review = col1.text_input("What do you think of the last flight you flew?", placeholder='e.g. This airline sucks 😫')

model = col2.selectbox(
    "Select Machine Learning Model", ['Multinomial Naive Bayes', 'Logistic', 'Linear SVM', 'ANN', 'RNN', 'LSTM']
)
col3.write("")
col3.write("")
if col3.button("Analyze"):
    if airline_review:
        prediction = predict_text(airline_review, model)
        st.write(prediction)

st.write("")
st.write("")
co1, co2, co3 = st.columns([3, 2, 1])
uploaded_file = co1.file_uploader("Upload reviews in a CSV file")
model = co2.selectbox(
    "Select Machine Learning Model for Files", ['Multinomial Naive Bayes', 'Logistic', 'Linear SVM', 'ANN', 'RNN', 'LSTM']
)
co3.write("")
co3.write("")
if co3.button('Analyze File'):
    if uploaded_file is not None:
        sentiment_data = pd.read_csv(uploaded_file)
        sentiment_data_preprocessed =  preprocess(sentiment_data)
        st.download_button('Download CSV', sentiment_data_preprocessed.to_csv(), 'text/csv')
    else:
        st.empty()

# if "screen_width" in st.session_state and st.session_state.screen_width < 768:
st.write("")
st.write("")

st.write("Here is an overview of all the models")
image = Image.open("output.png")
new_image = image.resize((1500, 400))
st.image(new_image)



@dataclass
class Component:
    name: str = None
    package: str = None
    demo: str = None
    forum_post: str = None
    github: str = None
    pypi: str = None
    image_url: str = None
    # screenshot_url: str = None
    stars: int = None
    github_description: str = None
    pypi_description: str = None
    avatar: str = None
    search_text: str = None
    github_author: str = None
    pypi_author: str = None
    created_at: datetime = None
    downloads: int = None
    categories: List[str] = None
