import streamlit as st
from recommender import content_based_recommendations
import pandas as pd

# Page config
st.set_page_config(page_title="📚 Book Recommender", layout="centered")

st.title("📖 Book Recommender System")
st.markdown("Get book suggestions using **Content-Based Filtering** !")

# Tabs for both recommenders
tab1, tab2 = st.tabs(["🔍 Recommend Books", "ℹ️ About Project"])

with tab1:
    st.subheader("🔍 Recommend by Book Description")
    book_data = pd.read_csv("data\\Book.csv")
    book_list = book_data["Book-Title"].dropna().unique().tolist()

    selected_book = st.selectbox("Select a Book:", sorted(book_list))

    if st.button("Recommend (Content-Based)"):
        with st.spinner("Finding similar books..."):
            recommendations = content_based_recommendations(selected_book)
        st.success("Recommendations:")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")

with tab2:
    import streamlit as st

    st.markdown("""
    # 📚 Book Recommender System  
    ### ✨ Built by *Mohd Inzamam*

    ---

    ### 🛠️ Project Overview:
    This intelligent Book Recommender System leverages **Content-Based Filtering** to provide personalized book suggestions based on users' reading preferences.

    Developed using **Python** and **Streamlit**, this app helps users discover new books tailored to their tastes by analyzing book content and user behavior.

    ---

    ### 🔍 Core Features:
    - **Content-Based Filtering**: Uses TF-IDF Vectorization and Cosine Similarity to recommend books similar to the selected title.
    - **Interactive UI**: Powered by Streamlit for seamless user experience.

    ---

    ### 🧠 How It Works:
    1. Processes book data using TF-IDF to extract textual features.
    2. Computes similarity between books for content-based recommendations.
    ---

    ### 📦 Tech Stack:
    - Python 3.10+
    - Streamlit
    - Pandas
    - Scikit-learn
    - NumPy

    ---

    ### 🌱 Future Enhancements:
    - Genre-based filtering
    - User profiles and ratings
    - Faster similarity search with libraries like FAISS
    - Visual analytics for user insights

    ---

    ### 👨🏻‍💻 Developed By:  
    **Mohd Inzamam**  

    > “Verily, knowledge is a light, and every good tool is a bridge to it.”  
    > May this recommender be a means of spreading beneficial knowledge, one book at a time. 📖

    ---
    """)
