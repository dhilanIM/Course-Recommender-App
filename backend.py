import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import streamlit as st

models = ("Course Similarity",
          #"User Profile",
          #"Clustering",
          "Clustering with PCA",
          #"KNN",
          #"NMF",
          #"Neural Network",
          #"Regression with Embedding Features", # Ridge
          #"Classification with Embedding Features" # Random forest
        )


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def load_course_genres():
    return pd.read_csv("course_genre.csv")


def normalize(df):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df

def load_user_profile():
    user_profile_df = pd.read_csv("user_profile.csv")
    #     features_names = list(user_profile_df.columns[1:])
    #     # Normalazing df
    #     user_profile_df[features_names] = normalize(user_profile_df[features_names])
    #     
    #     # user_profile_df[features_names] = scaler.fit_transform(user_profile_df[features_names])
    #     return user_profile_df
    
    return user_profile_df

def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def combine_cluster_labels(user_ids,labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df


def get_PCA(df,n_comps, only_transform=False):
    global model_pca
    model_pca = PCA(n_components=n_comps, random_state=1)
    if only_transform:
        return model_pca.transform(df)
    else:
        return model_pca.fit_transform(df)

# Model training
def train(model_name, params):
    # TODO: Add model training code here
    if model_name == models[0]:         # Similarity matrix doesn't need training process
        return None
    elif model_name == models[1]:       # PCA + Clustering
        features_df = load_user_profile()
        
        # Normalazing df
        features_names = list(features_df.columns[1:])
        features_df[features_names] = normalize(features_df[features_names])
        user_ids = features_df.loc[:, features_df.columns == 'user']
        features = features_df.loc[:, features_df.columns != 'user']

        # Performing PCA to reduce dimensions
        components = get_PCA(features,params['pca_no'])
        pca_df = pd.DataFrame(data=components)
        pca_df = user_ids.merge(pca_df, left_index=True, right_index=True)
        # pca_df.columns = ['user','PC0', 'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8']
        # pca_df.reset_index(drop=True)

        # Performing K-Means
        pca_features = pca_df.iloc[:,1:]
        kmeans = KMeans(n_clusters=params['cluster_no'], random_state=1)
        kmeans.fit(pca_features)
        # Combining cluster labels
        cluster_labels = kmeans.labels_
        global pca_cluster_df
        pca_cluster_df = combine_cluster_labels(user_ids,cluster_labels)
        return kmeans
    # elif model_name == models[2]:       # KNN w/ Surprise library
    #     rating_df = load_ratings()
        

# Prediction
def predict(model_name, user_ids, params, trained_model):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

        # TODO: Add prediction model code here
        elif model_name == models[1]:
            # Generating user profile
            ratings_df = load_ratings()
            genres_df = load_course_genres()
            courses_list = list(ratings_df[ratings_df['user'] == user_id]['item'].values)
            course_genre_df = genres_df[genres_df['COURSE_ID'].isin(courses_list)]
            course_genre_matrix = course_genre_df.iloc[:,2:].to_numpy()
            rated_courses = list(ratings_df[ratings_df['user'] == user_id]['rating'].values)
            current_user_profile = np.matmul(rated_courses,course_genre_matrix)
            current_user_profile = pd.DataFrame(current_user_profile.reshape(1,-1), columns=['Database','Python','CloudComputing','DataAnalysis','Containers','MachineLearning','ComputerVision','DataScience','BigData','Chatbot','R','BackendDev','FrontendDev','Blockchain'])

            # Concatenating current user profile
            features_df = load_user_profile()
            features = features_df.loc[:, features_df.columns != 'user']
            concat_df = pd.concat([features,current_user_profile],ignore_index=True)

            #Normalizing and doing PCA
            current_user_profile_norm_df = pd.DataFrame(normalize(concat_df))
            new_pca_df = pd.DataFrame(get_PCA(current_user_profile_norm_df,params['pca_no'], only_transform=False))
            current_user_profile_pca_df = new_pca_df.iloc[-1,:].values.reshape(1,-1)

            # Getting prediction
            predicted_label = trained_model.predict(current_user_profile_pca_df)

            # Getting recommended courses
            similar_users_df = pca_cluster_df[pca_cluster_df['cluster'].isin(predicted_label)]
            similar_users_ids = list(similar_users_df['user'].values)
            taken_courses = ratings_df[ratings_df['user'].isin(similar_users_ids)]['item']
            taken_courses_filtered = taken_courses[~taken_courses.isin(courses_list)]       #Filtering courses
            courses = list(taken_courses_filtered.value_counts().iloc[0:params['recommend_no']].index)
            users = [user_id]*len(courses)
            scores = [None]*len(courses)
           


    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
