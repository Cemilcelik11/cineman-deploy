import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the comprehensive movie dataset
@st.cache_data
def load_data():
    """
    Load the enhanced movie dataset with parental guide data, RT reviews, and plot summaries.
    This dataset provides comprehensive information for semantic matching and filtering.
    """
    df = pd.read_csv('movies_with_COMPLETE_parental_guide_FIXED.csv')
    return df

# Load the embedding model for semantic search
@st.cache_resource
def load_model():
    """
    Load the sentence transformer model for creating semantic embeddings.
    Uses the all-MiniLM-L6-v2 model for efficient and accurate text representations.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load pre-computed embeddings for instant semantic search
@st.cache_data
def load_precomputed_embeddings():
    """
    Load pre-computed embeddings and comprehensive texts from cache files.
    This enables instant semantic search without real-time embedding generation.
    """
    embeddings_dir = 'embeddings_cache'
    
    # Define cache file paths
    embeddings_file = os.path.join(embeddings_dir, 'movie_embeddings.npy')
    texts_file = os.path.join(embeddings_dir, 'comprehensive_texts.pkl')
    metadata_file = os.path.join(embeddings_dir, 'metadata.json')
    
    # Verify cache files exist
    if not all(os.path.exists(f) for f in [embeddings_file, texts_file, metadata_file]):
        st.error("Pre-computed embeddings not found. Please run the pre-computation script first.")
        st.info("Run: python precompute_embeddings.py")
        st.stop()
    
    # Load pre-computed embeddings
    embeddings = np.load(embeddings_file)
    
    # Load comprehensive text representations
    with open(texts_file, 'rb') as f:
        comprehensive_texts = pickle.load(f)
    
    # Load metadata for verification and display
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return embeddings, comprehensive_texts, metadata

# Legacy embedding function for fallback scenarios
@st.cache_data
def create_embeddings(texts, _model):
    """
    Create embeddings for text content. This function serves as a fallback
    when pre-computed embeddings are not available.
    """
    processed_texts = [str(text) if not pd.isna(text) else "" for text in texts]
    return _model.encode(processed_texts)

def decode_json_reviews(json_string):
    """
    Safely decode JSON review strings from Rotten Tomatoes data.
    Handles various edge cases in the stored review data format.
    """
    if pd.isna(json_string) or json_string == '' or json_string == '[]':
        return []
    
    try:
        if isinstance(json_string, str):
            # Clean up common JSON encoding issues
            cleaned = json_string.replace("\\'", "'").replace('\\"', '"')
            reviews = json.loads(cleaned)
            return reviews if isinstance(reviews, list) else []
        return []
    except (json.JSONDecodeError, TypeError):
        return []

def generate_comprehensive_text(row):
    """
    Generate comprehensive text combining plot summary, user reviews, and critic reviews.
    This creates rich text representations for enhanced semantic matching capabilities.
    """
    text_parts = []
    
    # Include plot summary as primary narrative content
    if pd.notna(row.get('plot_summary')):
        plot_text = str(row['plot_summary']).strip()
        if plot_text and plot_text not in ['N/A', 'No Wikipedia page found for this movie']:
            text_parts.append(plot_text)
    
    # Include user reviews for audience perspective and sentiment
    if pd.notna(row.get('rt_user_reviews')):
        try:
            user_reviews = decode_json_reviews(row['rt_user_reviews'])
            if user_reviews and len(user_reviews) > 0:
                # Sample first few user reviews for audience perspective
                user_sample = user_reviews[:3]
                user_text = " ".join(user_sample)
                if len(user_text) > 50:
                    text_parts.append(user_text[:300])
        except Exception:
            pass
    
    # Include critic reviews for professional analysis and themes
    if pd.notna(row.get('rt_critic_reviews')):
        try:
            critic_reviews = decode_json_reviews(row['rt_critic_reviews'])
            if critic_reviews and len(critic_reviews) > 0:
                # Sample first few critic reviews for professional perspective
                critic_sample = critic_reviews[:2]
                critic_text = " ".join(critic_sample)
                if len(critic_text) > 50:
                    text_parts.append(critic_text[:250])
        except Exception:
            pass
    
    # Return combined comprehensive text or fallback to title
    return " ".join(text_parts) if text_parts else str(row.get('title', ''))

def main():
    """
    Main application function containing the complete movie recommendation interface.
    Provides advanced filtering capabilities and semantic search functionality.
    """
    st.title("Movie Recommendation System")
    st.markdown("*Discover movies through intelligent search and comprehensive filtering*")
    
    # Load dataset and initialize models
    df = load_data()
    model = load_model()
    
    # Load pre-computed embeddings for enhanced performance
    try:
        all_embeddings, all_texts, metadata = load_precomputed_embeddings()
        embeddings_loaded = True
    except:
        embeddings_loaded = False
        st.warning("Pre-computed embeddings not available. Search will use real-time processing.")
    
    # Display dataset statistics and performance information
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.markdown(f"**Total movies:** {len(df):,}")
    
    if embeddings_loaded:
        st.sidebar.markdown(f"**Pre-computed embeddings:** ✅")
        st.sidebar.markdown(f"**Instant search ready:** ⚡")
        st.sidebar.markdown(f"**Last updated:** {metadata['creation_timestamp'][:10]}")
    else:
        st.sidebar.markdown(f"**Search mode:** Real-time processing")
    
    # Create comprehensive filter interface
    st.sidebar.header("Filter Options")
    
    # Year filter with expandable interface
    with st.sidebar.expander("Year Filter", expanded=False):
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        year_range = st.slider(
            "Select Year Range",
            min_year, max_year,
            (min_year, max_year)
        )
    
    # Genre filter with multi-selection capability
    with st.sidebar.expander("Genre Filter", expanded=False):
        all_genres = set()
        for genres in df['genres'].dropna().str.split(', '):
            if isinstance(genres, list):
                all_genres.update(genres)
        selected_genre = st.multiselect(
            "Select Genre(s)",
            sorted(list(all_genres))
        )
    
    # Runtime filter for movie length preferences
    with st.sidebar.expander("Runtime Filter", expanded=False):
        min_runtime = int(df['runtime'].min())
        max_runtime = int(df['runtime'].max())
        runtime_range = st.slider(
            "Select Runtime Range (minutes)",
            min_runtime, max_runtime,
            (min_runtime, max_runtime)
        )
    
    # Director filter for filmmaker preferences
    with st.sidebar.expander("Director Filter", expanded=False):
        directors = sorted(df['directors'].dropna().unique())
        selected_directors = st.multiselect("Select Director(s)", directors)
    
    # Cast filter for actor preferences
    with st.sidebar.expander("Actor Filter", expanded=False):
        all_actors = set()
        for actors in df['cast'].dropna().str.split(', '):
            if isinstance(actors, list):
                all_actors.update(actors)
        selected_actors = st.multiselect("Select Actor(s)", sorted(list(all_actors)))
    
    # Language filter for international cinema
    with st.sidebar.expander("Language Filter", expanded=False):
        languages = sorted(df['languages'].str.split(', ').explode().dropna().unique())
        selected_language = st.selectbox("Select Language", [""] + languages)
    
    # Country filter for regional preferences
    with st.sidebar.expander("Country Filter", expanded=False):
        countries = sorted(df['countries'].str.split(', ').explode().dropna().unique())
        selected_country = st.selectbox("Select Country", [""] + countries)
    
    # Rating filters for quality thresholds
    with st.sidebar.expander("Rating Filters", expanded=False):
        min_imdb = st.slider(
            "Minimum IMDB Rating",
            0.0, 10.0, 0.0, 0.1
        )
    
    # Advanced content rating filters using parental guide data
    with st.sidebar.expander("Content Rating Filters", expanded=False):
        st.markdown("*Based on detailed parental guide analysis*")
        
        # Define content rating categories with severity levels
        rating_codes = [
            ('violence_severity', 'Violence'), 
            ('profanity_severity', 'Profanity'), 
            ('sex_nudity_severity', 'Sexual Content'), 
            ('alcohol_drugs_severity', 'Drug Use'), 
            ('frightening_severity', 'Intense Scenes')
        ]
        
        code_filters = {}
        
        for code, label in rating_codes:
            if code in df.columns:
                min_code = float(df[code].min())
                max_code = float(df[code].max())
                code_range = st.slider(
                    f"{label} Range (0-5)",
                    min_code, max_code,
                    (min_code, max_code),
                    help=f"0=None, 5=Most Intense"
                )
                code_filters[code] = code_range
    
    # Main search interface
    st.header("Describe the movie you're looking for")
    st.markdown("*Use natural language to describe plot, themes, mood, or any movie characteristics*")
    
    search_text = st.text_area(
        "Enter description (optional)", 
        placeholder="Example: A thrilling mafia story with family drama and great acting...",
        height=100
    )
    
    # Results configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort Results By",
            ["Relevance (if search used)", "IMDB Rating", "Release Year (newest)", "Runtime"]
        )
    
    with col2:
        results_limit = st.selectbox("Results Limit", [5, 10, 20, 30, 50, 100])
    
    # Search execution
    search_clicked = st.button("🔍 Search Movies", type="primary")
    
    # Process search request when button is activated
    if search_clicked:
        with st.spinner("Processing your search..." if not embeddings_loaded else "Searching movies..."):
            # Initialize filtered dataset
            filtered_df = df.copy()
            
            # Apply year range filter
            if year_range != (min_year, max_year):
                filtered_df = filtered_df[filtered_df['year'].between(year_range[0], year_range[1])]
            
            # Apply runtime filter
            if runtime_range != (min_runtime, max_runtime):
                filtered_df = filtered_df[filtered_df['runtime'].between(runtime_range[0], runtime_range[1])]
            
            # Apply IMDB rating filter
            if min_imdb > 0:
                filtered_df = filtered_df[filtered_df['imdb_rating'] >= min_imdb]
            
            # Apply genre filter
            if selected_genre:
                filtered_df = filtered_df[filtered_df['genres'].apply(
                    lambda x: any(genre in str(x).split(', ') for genre in selected_genre)
                )]
            
            # Apply director filter
            if selected_directors:
                filtered_df = filtered_df[filtered_df['directors'].apply(
                    lambda x: any(director in str(x) for director in selected_directors)
                )]
            
            # Apply cast filter
            if selected_actors:
                filtered_df = filtered_df[filtered_df['cast'].apply(
                    lambda x: any(actor in str(x) for actor in selected_actors)
                )]
            
            # Apply language filter
            if selected_language:
                filtered_df = filtered_df[filtered_df['languages'].str.contains(selected_language, case=False, na=False)]
            
            # Apply country filter
            if selected_country:
                filtered_df = filtered_df[filtered_df['countries'].str.contains(selected_country, case=False, na=False)]
            
            # Apply content rating filters using parental guide severity data
            for code, code_range in code_filters.items():
                min_val, max_val = code_range
                min_code = float(df[code].min())
                max_code = float(df[code].max())
                
                if (min_val, max_val) != (min_code, max_code):
                    filtered_df = filtered_df[
                        (filtered_df[code].isna()) | 
                        (filtered_df[code].between(min_val, max_val))
                    ]
            
            # Execute semantic search if search text is provided
            if search_text.strip():
                if len(filtered_df) > 0:
                    if embeddings_loaded:
                        # Use pre-computed embeddings for instant search
                        filtered_indices = filtered_df.index.tolist()
                        filtered_embeddings = all_embeddings[filtered_indices]
                        query_embedding = model.encode([search_text])
                        similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
                        
                        # Apply similarity scores and sort results
                        filtered_df = filtered_df.copy()
                        filtered_df['similarity'] = similarities
                        filtered_df = filtered_df.sort_values('similarity', ascending=False)
                        
                        st.success(f"⚡ Instant semantic search completed! Found {len(filtered_df)} relevant movies.")
                    else:
                        # Fallback to real-time embedding generation
                        comprehensive_texts = [generate_comprehensive_text(row) for _, row in filtered_df.iterrows()]
                        plot_embeddings = create_embeddings(comprehensive_texts, model)
                        query_embedding = model.encode([search_text])
                        similarities = cosine_similarity(query_embedding, plot_embeddings)[0]
                        
                        filtered_df = filtered_df.copy()
                        filtered_df['similarity'] = similarities
                        filtered_df = filtered_df.sort_values('similarity', ascending=False)
                        
                        st.success(f"🎯 Semantic search completed! Found {len(filtered_df)} relevant movies.")
                else:
                    st.warning("No movies match your filters. Try broadening your search criteria.")
            else:
                if len(filtered_df) > 0:
                    st.info(f"📊 Filter applied! Found {len(filtered_df)} movies matching your criteria.")
                else:
                    st.warning("No movies match your filters. Try broadening your search criteria.")
            
            # Apply sorting preferences if not using semantic search relevance
            if not search_text.strip() or sort_by != "Relevance (if search used)":
                if sort_by == "IMDB Rating":
                    filtered_df = filtered_df.sort_values('imdb_rating', ascending=False)
                elif sort_by == "Release Year (newest)":
                    filtered_df = filtered_df.sort_values('year', ascending=False)
                elif sort_by == "Runtime":
                    filtered_df = filtered_df.sort_values('runtime', ascending=False)
            
            # Limit results to specified number
            filtered_df = filtered_df.head(results_limit)
            
            # Display search results with comprehensive movie information
            if len(filtered_df) > 0:
                st.header(f"🎬 Found {len(filtered_df)} Movies")
                
                for _, movie in filtered_df.iterrows():
                    # Movie title and year header
                    title = movie.get('title', 'Unknown Title')
                    year = movie.get('year', 'Unknown')
                    st.subheader(f"{title} ({int(year)})")
                    
                    # Two-column layout for movie information
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Plot summary and narrative information
                        plot = movie.get('plot_summary', 'No plot summary available')
                        if pd.notna(plot) and str(plot).strip() and plot not in ['N/A', 'No Wikipedia page found for this movie']:
                            st.write(f"**Plot:** {plot}")
                        
                        # Director information
                        director = movie.get('directors', 'Unknown')
                        if pd.notna(director):
                            st.write(f"**Director:** {director}")
                        
                        # Cast information
                        cast = movie.get('cast', 'Unknown')
                        if pd.notna(cast):
                            st.write(f"**Cast:** {cast}")
                        
                        # Genre classification
                        genres = movie.get('genres', 'Unknown')
                        if pd.notna(genres):
                            st.write(f"**Genres:** {genres}")
                        
                        # Language and country information
                        languages = movie.get('languages', '')
                        if pd.notna(languages) and languages:
                            st.write(f"**Languages:** {languages}")
                        
                        countries = movie.get('countries', '')
                        if pd.notna(countries) and countries:
                            st.write(f"**Countries:** {countries}")
                    
                    with col2:
                        # Ratings and quality metrics
                        if pd.notna(movie.get('imdb_rating')):
                            st.write(f"**IMDB Rating:** {movie['imdb_rating']:.1f}/10")
                        
                        # Runtime information
                        if pd.notna(movie.get('runtime')):
                            st.write(f"**Runtime:** {int(movie['runtime'])} minutes")
                        
                        # Search relevance score if semantic search was used
                        if search_text.strip() and 'similarity' in movie:
                            st.write(f"**Search Relevance:** {movie['similarity']:.2f}")
                        
                        # Content rating information from parental guide data
                        content_ratings = []
                        severity_columns = [
                            ('violence_severity', 'Violence'), 
                            ('profanity_severity', 'Language'), 
                            ('sex_nudity_severity', 'Sexual Content'), 
                            ('alcohol_drugs_severity', 'Substances'),
                            ('frightening_severity', 'Intense Scenes')
                        ]
                        
                        for col, label in severity_columns:
                            if col in movie and pd.notna(movie[col]) and movie[col] > 0:
                                severity_level = int(movie[col])
                                severity_labels = {1: 'Mild', 2: 'Moderate', 3: 'Strong', 4: 'Intense', 5: 'Severe'}
                                severity_text = severity_labels.get(severity_level, f'Level {severity_level}')
                                content_ratings.append(f"{label}: {severity_text}")
                        
                        if content_ratings:
                            st.write(f"**Content:** {', '.join(content_ratings)}")
                        else:
                            st.write("**Content:** Family Friendly")
                        
                        # MPAA rating if available
                        if pd.notna(movie.get('mpaa_rating')):
                            st.write(f"**MPAA Rating:** {movie['mpaa_rating']}")
                        
                        # External links for additional information
                        imdb_id = movie.get('imdb_id', '')
                        if pd.notna(imdb_id) and str(imdb_id).strip():
                            imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
                            st.markdown(f"[🎬 View on IMDb]({imdb_url})")
                    
                    # Optional review preview section
                    user_reviews = decode_json_reviews(movie.get('rt_user_reviews', ''))
                    critic_reviews = decode_json_reviews(movie.get('rt_critic_reviews', ''))
                    
                    if user_reviews or critic_reviews:
                        total_reviews = len(user_reviews) + len(critic_reviews)
                        with st.expander(f"📝 Sample Reviews ({total_reviews} available)", expanded=False):
                            if user_reviews:
                                st.write("**User Reviews:**")
                                for review in user_reviews[:2]:
                                    st.write(f"• {review}")
                                if len(user_reviews) > 2:
                                    st.write(f"*...and {len(user_reviews) - 2} more user reviews*")
                            
                            if critic_reviews:
                                if user_reviews:
                                    st.write("")
                                st.write("**Critic Reviews:**")
                                for review in critic_reviews[:2]:
                                    st.write(f"• {review}")
                                if len(critic_reviews) > 2:
                                    st.write(f"*...and {len(critic_reviews) - 2} more critic reviews*")
                    
                    st.divider()
            
            else:
                # No results found - provide helpful suggestions
                st.warning("🔍 No movies found matching your criteria.")
                st.markdown("**Try these suggestions:**")
                st.markdown("• Broaden your filter selections")
                st.markdown("• Use different search terms")
                st.markdown("• Check your spelling and try simpler descriptions")
                st.markdown("• Reduce the number of active filters")

    # Application footer with helpful information
    st.markdown("---")
    st.markdown("### 💡 Search Tips")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Describe Themes:**")
        st.markdown("*'Coming of age story with friendship themes'*")
    
    with col2:
        st.markdown("**Describe Mood:**")
        st.markdown("*'Light-hearted comedy with witty dialogue'*")
    
    with col3:
        st.markdown("**Describe Elements:**")
        st.markdown("*'Sci-fi with time travel and great visuals'*")

if __name__ == "__main__":
    main()