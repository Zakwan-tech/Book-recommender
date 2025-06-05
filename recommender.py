import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
book_data = pd.read_csv("data/Book.csv")

book_data['Book-Title'] = book_data['Book-Title'].fillna("")

book_data['Book-Profile'] = book_data['Book-Profile'].fillna("")

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(book_data['Book-Title'])

def content_based_recommendations(book_title, top_n=10):
    # Get index of the book
    idx = book_data[book_data['Book-Title'].str.lower() == book_title.lower()].index
    if idx.empty:
        return ["Book not found."]
    idx = idx[0]

    # Compute similarity of the selected book with all others
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get indices of top similar books (excluding self)
    sim_scores = cosine_sim.argsort()[-top_n-1:-1][::-1]
    recommended_books = book_data.iloc[sim_scores]['Book-Title'].values

    return recommended_books
