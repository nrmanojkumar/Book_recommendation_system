#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fuzzywuzzy import process

# Load your dataset
df = pd.read_csv('updated_books.csv')

# Fill missing ISBN 
df['isbn'] = df['isbn'].fillna('NA')

df["original_publication_year"].fillna(method="ffill", inplace= True)

most_common_language = df['language_code'].mode()[0]
df['language_code'] = df['language_code'].fillna(most_common_language)

df.drop(columns= ["original_title"], axis=1, inplace=True)

# Combine the 'authors', 'title', 'original_publication_year', 'average_rating', and 'ratings_count' columns into a single column
df['content'] = df['authors'] + " " + df['title'] + " " + df['original_publication_year'].astype(str) + " " + df['average_rating'].astype(str) + " " + df['ratings_count'].astype(str)

# Create a TF-IDF vectorizer and fit_transform the 'content' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute the cosine similarity matrix from the TF-IDF vectors
cosine_sim_content = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Convert the cosine similarity matrix into a DataFrame
cosine_sim_content_df = pd.DataFrame(cosine_sim_content, index=df['book_id'], columns=df['book_id'])

# Your existing collaborative filtering code
pivot_table = df.pivot_table(index='book_id', columns='average_rating', values='ratings_count').fillna(0)
cosine_sim = cosine_similarity(pivot_table, pivot_table)
cosine_sim_df = pd.DataFrame(cosine_sim, index=pivot_table.index, columns=pivot_table.index)

# Add up the two similarity score matrices
hybrid_sim_df = cosine_sim_df + cosine_sim_content_df

def get_recommendations(book_input, N):
    # Check if the input is a digit (book_id)
    if book_input.isdigit():
        book_id = int(book_input)
    else:
        # If the input is not a digit, use fuzzy matching to find the closest match in the dataset
        match = process.extractOne(book_input, df['title'].tolist() + df['authors'].tolist())
        book_id = df[(df['title'] == match[0]) | (df['authors'] == match[0])]['book_id'].values[0]

    # Get the row corresponding to the book_id
    book_similarities = hybrid_sim_df.loc[book_id]

    # Get the top N most similar books
    top_N_books = book_similarities.nlargest(N + 1).iloc[1:]

    # Get the titles of the top N books
    top_N_titles = df[df['book_id'].isin(top_N_books.index)]['title']

    return top_N_titles

# Streamlit code
st.title('Book Recommendation System')
book_input = st.text_input("Please enter a book ID, title or author: ")
N = st.slider("Please enter the number of recommendations you want: ", 1, 10)
if st.button('Get Recommendations'):
    recommendations = get_recommendations(book_input, N)
    st.write(recommendations)

