import streamlit as st
import pandas as pd
import pickle

# ==========================================
# 1. LOAD PRECOMPUTED PICKLE FILES
# ==========================================
@st.cache_data  # Keeps data in memory so it doesn't reload on every click
def load_data():
    try:
        # Load your cleaned dataframe (6 columns)
        with open('df.pkl', 'rb') as f:
            movies_df = pickle.load(f)
            
        # Load your precomputed similarity data / indices mapping
        with open('indices.pkl', 'rb') as f:
            similarity_data = pickle.load(f)
            
        # Ensure title strings don't have trailing whitespace
        movies_df['title'] = movies_df['title'].astype(str).str.strip()
        
        return movies_df, similarity_data
    except FileNotFoundError as e:
        st.error(f"Missing pickle file error: {e}. Ensure 'df.pkl' and 'indices.pkl' are in the root directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading pickle files: {e}")
        st.stop()

# Initialize files
movies, similarity_indices = load_data()

# ==========================================
# 2. RECOMMENDATION ALGORITHM LOGIC
# ==========================================
def get_recommendations(movie_title, top_n=5):
    try:
        # 1. Find the index of the selected movie title
        if movie_title not in movies['title'].values:
            st.warning(f"'{movie_title}' not found in dataset mapping.")
            return pd.DataFrame()
            
        movie_idx = movies[movies['title'] == movie_title].index[0]
        
        # 2. Fetch precomputed target matches using your indices array
        # This handles cases where similarity_indices is a matrix array or a dictionary list
        if hasattr(similarity_indices, 'ndim') and similarity_indices.ndim == 2:
            # If it's a 2D matrix, extract the row matching the current movie index
            # Sort scores and grab top matching items (skipping the first item as it's the movie itself)
            sim_scores = list(enumerate(similarity_indices[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_movie_records = sim_scores[1:top_n+1]
            target_indices = [i[0] for i in top_movie_records]
        else:
            # Fallback if indices.pkl is already a precomputed mapping slice list
            target_indices = similarity_indices[movie_idx][1:top_n+1]
        
        # 3. Safe check bounds and slice dataframe
        return movies.iloc[target_indices]
        
    except Exception as e:
        st.error(f"Recommendation generation pipeline error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. STREAMLIT INTERACTIVE USER INTERFACE
# ==========================================

# Page Tab Config
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# App Title & Header Banner
st.title("🎬 CineMatch: NLP Movie Recommendation System")
st.markdown("""
Welcome to the production deployment of **CineMatch**. Select a movie below, and our precomputed NLP pipeline will instantly surface contextually similar films.
""")
st.write("---")

# Sidebar Dynamic Settings Layout
st.sidebar.header("⚙️ Filter Configurations")
num_recommendations = st.sidebar.slider(
    "Total Movies to Recommend:", 
    min_value=3, 
    max_value=10, 
    value=5
)

st.sidebar.markdown("""
---
### System Pipeline Info
* **Corpus Size:** 45,453 unique movies
* **Vector Model:** TF-IDF Features + Cosine Similarity mapping
* **NLP Package:** NLTK text processing
""")

# Main Content Layout Split
col1, col2 = st.columns([2, 1])

with col1:
    movie_titles = movies['title'].unique()
    selected_movie = st.selectbox(
        "🍿 Choose or Type a Movie Title:",
        options=movie_titles,
        index=0,
        help="Type or scroll to select from the dataset catalog."
    )

with col2:
    st.markdown("### Selected Movie Profile")
    # Lookup values safely for selected entry
    selected_row = movies[movies['title'] == selected_movie].iloc[0]
    
    # Handle NaN values smoothly using fallback text defaults
    tagline_text = selected_row['tagline'] if pd.notna(selected_row['tagline']) else "No tagline available."
    rating = selected_row['vote_average'] if pd.notna(selected_row['vote_average']) else "N/A"
    
    st.markdown(f"**Score:** ⭐ `{rating}/10`  |  **Popularity:** 📈 `{round(float(selected_row['popularity']), 2)}`")
    st.markdown(f"*\"{tagline_text}\"*")

st.write("")

# Execution Button Core
if st.button("🚀 Find Similar Movies", type="primary"):
    with st.spinner("Traversing high-dimensional vector similarities..."):
        
        # Run recommendation calculations
        results = get_recommendations(selected_movie, top_n=num_recommendations)
        
        if not results.empty:
            st.success(f"Successfully retrieved top matches for **{selected_movie}**:")
            st.write("")
            
            # Dynamically draw structured UI grid column segments for results
            cols = st.columns(len(results))
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i]:
                    st.markdown(f"#### {row['title']}")
                    
                    # Rating chip
                    row_rating = row['vote_average'] if pd.notna(row['vote_average']) else "N/A"
                    st.caption(f"⭐ Score: `{row_rating}/10`")
                    
                    # Overview truncation logic so visual card height remains even
                    overview_snippet = row['overview'] if pd.notna(row['overview']) else "No synopsis description found."
                    if len(overview_snippet) > 130:
                        overview_snippet = overview_snippet[:130] + "..."
                        
                    st.write(f"*{overview_snippet}*")
        else:
            st.warning("Empty matrix match. Confirm matrix integrity or indexing shapes inside indices.pkl.")

# Persistent UI Footer
st.write("---")
st.caption("Powered by Python, Streamlit Engine, and Scikit-Learn Vector Architecture.")
