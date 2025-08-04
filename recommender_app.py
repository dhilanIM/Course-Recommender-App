import streamlit as st
import pandas as pd
import time
import backend as backend
from sklearn.cluster import KMeans

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()

@st.cache_data
def load_user_profile():
    return backend.load_user_profile()


@st.cache_data
def load_course_genre():
    return backend.load_course_genres()


# Initialize the app by first loading datasets
def init__recommender_app():
    st.header("📚 Course Recommender System")
    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()
        user_profile_df = load_user_profile()
        curse_genre_df = load_course_genre()
    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, params=None):

    if model_name == backend.models[0]:
        # Start training course similarity model
        with st.spinner('Training...'):
            time.sleep(0.5)
            return backend.train(model_name,params)
        st.success('Done!')
    elif model_name == backend.models[1]:
        with st.spinner('Training...'):
            time.sleep(0.5)
        return backend.train(model_name,params)
        st.success('Done!')
    # elif model_name == backend.models[2]:
    #     with st.spinner('Training...'):
    #         time.sleep(0.5)
    #     return backend.train(model_name,params)
    #     st.success('Done!')

def predict(model_name, user_ids, params, trained_model = None):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params, trained_model)
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar

# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# TODO: Add hyper-parameters for other models

# User Cluster with PCA
elif model_selection == backend.models[1]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
    pca_no = st.sidebar.slider('Number of components (For PCA)',
                                   min_value=4, max_value=13,
                                   value=8, step=1)
    recommend_no = st.sidebar.slider('Number of recommendations',
                                   min_value=1, max_value=6,
                                   value=3, step=1)
    params['cluster_no'] = cluster_no
    params['pca_no'] = pca_no
    params['recommend_no'] = recommend_no
    
# # KNN
# elif model_selection == backend.models[2]:
#     neighbors_no = st.sidebar.slider('Number of Neighbors',
#                                    min_value=0, max_value=20,
#                                    value=10, step=1)
#     params['neighbors_no'] = neighbors_no

# # NMF
# elif model_selection == backend.models[3]:
#     init_low = st.sidebar.slider('Initial lower bound',
#                                    min_value=0.1, max_value=0.6,
#                                    value=0.5, step=0.1)
#     init_high = st.sidebar.slider('Initial higher bound',
#                                    min_value=1.0, max_value=1.8,
#                                    value=1.5, step=0.1)   
#     factors_no = st.sidebar.slider('Number of factors to use',
#                                    min_value=25, max_value=40,
#                                    value=32, step=1)
#     params['init_low'] = init_low
#     params['init_high'] = init_high
#     params['factors_no'] = factors_no
    
# # Regression with Embedding Features  (Ridge)  
# elif model_selection == backend.models[4]:
#     shrinkage_amount = st.sidebar.slider('Shrinkage amount (alpha)',
#                                    min_value=0.1, max_value=0.8,
#                                    value=0.2, step=0.1)
#     params['shrinkage_amount'] = shrinkage_amount
    
# # Classification with Embedding Features  (Random forest)
# elif model_selection == backend.models[5]:
#     max_depth = st.sidebar.slider('Maximum Depth',
#                                    min_value=74, max_value=120,
#                                    value=100, step=1)
#     params['max_depth'] = max_depth           
                                


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if 'trained' not in st.session_state:
    st.session_state.trained = None


if training_button:
    # global trained_model
    trained_model = train(model_selection, params)
    st.session_state.trained = trained_model

# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params, trained_model=st.session_state.trained)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
    st.dataframe(res_df)
