import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
from PIL import Image
import httpx
import pypistats
import requests
import streamlit as st
import yaml
from bs4 import BeautifulSoup
from markdownlit import mdlit
from stqdm import stqdm

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


icon("🎪")
"""
# Sentiment Analysis

[![](https://img.shields.io/github/stars/jrieke/components-hub?style=social)](https://github.com/jrieke/components-hub) &nbsp; [![](https://img.shields.io/twitter/follow/jrieke?style=social)](https://twitter.com/jrieke)
"""


description_text = """
Discover {} Streamlit components! Most information on this page is 
automatically crawled from Github, PyPI, and the 
[Streamlit forum](https://discuss.streamlit.io/t/streamlit-components-community-tracker/4634).
If you build your own [custom component](https://docs.streamlit.io/library/components/create), 
it should appear here within a few days.
"""
description = st.empty()
description.write(description_text.format("all"))
col1, col2, col3 = st.columns([3, 2, 1])
airline_review = col1.text_input("Enter whatever Text you want", placeholder='e.g. "image" or "text" or "card"')

model = col2.selectbox(
    "Select Machine Learning Model", ['Multinomial Naive Bayes', 'Logistic', 'Linear SVM', 'ANN', 'RNN', 'LSTM']
)
col3.write("")
col3.write("")
if col3.button("Analyze"):
    if airline_review:
        if model == "Multinomial Naive Bayes":
            st.write("This is a positive review")
        elif model == "Logistic":
            st.write("This is a negative review")


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
