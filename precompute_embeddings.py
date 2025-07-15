#!/usr/bin/env python3
"""
Pre-compute Movie Embeddings Script
Generates comprehensive text representations and embeddings for the entire movie dataset
"""

import pandas as pd
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from datetime import datetime

def decode_json_reviews(json_string):
    """Safely decode JSON review strings from RT data"""
    if pd.isna(json_string) or json_string == '' or json_string == '[]':
        return []
    
    try:
        if isinstance(json_string, str):
            cleaned = json_string.replace("\\'", "'").replace('\\"', '"')
            reviews = json.loads(cleaned)
            return reviews if isinstance(reviews, list) else []
        return []
    except (json.JSONDecodeError, TypeError):
        return []

def generate_comprehensive_text(row):
    """Generate comprehensive text combining plot summary, user reviews, and critic reviews"""
    text_parts = []
    
    # Use plot_summary as the primary content
    if pd.notna(row.get('plot_summary')):
        plot_text = str(row['plot_summary']).strip()
        if plot_text and plot_text not in ['N/A', 'No Wikipedia page found for this movie']:
            text_parts.append(plot_text)
    
    # Add user reviews if available
    if pd.notna(row.get('rt_user_reviews')):
        try:
            user_reviews = decode_json_reviews(row['rt_user_reviews'])
            if user_reviews and len(user_reviews) > 0:
                user_sample = user_reviews[:3]
                user_text = " ".join(user_sample)
                if len(user_text) > 50:
                    text_parts.append(user_text[:300])
        except Exception:
            pass
    
    # Add critic reviews if available
    if pd.notna(row.get('rt_critic_reviews')):
        try:
            critic_reviews = decode_json_reviews(row['rt_critic_reviews'])
            if critic_reviews and len(critic_reviews) > 0:
                critic_sample = critic_reviews[:2]
                critic_text = " ".join(critic_sample)
                if len(critic_text) > 50:
                    text_parts.append(critic_text[:250])
        except Exception:
            pass
    
    # Return combined text or fallback to title
    return " ".join(text_parts) if text_parts else str(row.get('title', ''))

def create_embeddings_for_dataset(csv_file_path, output_dir='embeddings_cache'):
    """
    Main function to create and save embeddings for the entire dataset
    """
    print("="*60)
    print("MOVIE DATASET EMBEDDING PRE-COMPUTATION")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded {len(df):,} movies")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Load the embedding model
    print("Loading SentenceTransformer model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Generate comprehensive texts for all movies
    print("Generating comprehensive text representations...")
    comprehensive_texts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing movies"):
        comprehensive_text = generate_comprehensive_text(row)
        comprehensive_texts.append(comprehensive_text)
    
    print(f"Generated {len(comprehensive_texts)} text representations")
    
    # Create embeddings
    print("Creating embeddings (this may take several minutes)...")
    try:
        # Process in batches to manage memory efficiently
        batch_size = 100
        all_embeddings = []
        
        for i in tqdm(range(0, len(comprehensive_texts), batch_size), desc="Creating embeddings"):
            batch_texts = comprehensive_texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings)
        print(f"Successfully created embeddings with shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return False
    
    # Save the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save embeddings as numpy array
    embeddings_file = os.path.join(output_dir, 'movie_embeddings.npy')
    np.save(embeddings_file, embeddings)
    print(f"Embeddings saved to: {embeddings_file}")
    
    # Save comprehensive texts
    texts_file = os.path.join(output_dir, 'comprehensive_texts.pkl')
    with open(texts_file, 'wb') as f:
        pickle.dump(comprehensive_texts, f)
    print(f"Comprehensive texts saved to: {texts_file}")
    
    # Save metadata for verification
    metadata = {
        'dataset_file': csv_file_path,
        'total_movies': len(df),
        'embeddings_shape': embeddings.shape,
        'model_name': 'all-MiniLM-L6-v2',
        'creation_timestamp': timestamp,
        'embedding_dimension': embeddings.shape[1]
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")
    
    # Generate summary report
    print("\n" + "="*60)
    print("PRE-COMPUTATION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Dataset: {len(df):,} movies processed")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {embeddings_file}")
    print(f"  - {texts_file}")
    print(f"  - {metadata_file}")
    
    # Analyze text quality
    text_lengths = [len(text) for text in comprehensive_texts]
    avg_length = np.mean(text_lengths)
    print(f"\nText Quality Analysis:")
    print(f"  Average text length: {avg_length:.0f} characters")
    print(f"  Minimum text length: {min(text_lengths)} characters")
    print(f"  Maximum text length: {max(text_lengths)} characters")
    
    # Count movies with different content types
    with_plot = sum(1 for _, row in df.iterrows() if pd.notna(row.get('plot_summary')))
    with_user_reviews = sum(1 for _, row in df.iterrows() if decode_json_reviews(row.get('rt_user_reviews', '')))
    with_critic_reviews = sum(1 for _, row in df.iterrows() if decode_json_reviews(row.get('rt_critic_reviews', '')))
    
    print(f"\nContent Coverage:")
    print(f"  Movies with plot summaries: {with_plot:,} ({with_plot/len(df)*100:.1f}%)")
    print(f"  Movies with user reviews: {with_user_reviews:,} ({with_user_reviews/len(df)*100:.1f}%)")
    print(f"  Movies with critic reviews: {with_critic_reviews:,} ({with_critic_reviews/len(df)*100:.1f}%)")
    
    print(f"\nYour movie recommendation system is now ready for high-performance semantic search!")
    return True

if __name__ == "__main__":
    # Configuration
    DATASET_FILE = 'movies_with_COMPLETE_parental_guide_FIXED.csv'
    OUTPUT_DIRECTORY = 'embeddings_cache'
    
    print("Movie Dataset Embedding Pre-Computation")
    print("This script will create embeddings for your entire movie dataset.")
    print(f"Dataset file: {DATASET_FILE}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    
    # Ask for confirmation
    response = input("\nProceed with embedding generation? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        success = create_embeddings_for_dataset(DATASET_FILE, OUTPUT_DIRECTORY)
        if success:
            print("\n✅ Pre-computation completed successfully!")
            print("You can now update your Streamlit app to use the pre-computed embeddings.")
        else:
            print("\n❌ Pre-computation failed. Please check the error messages above.")
    else:
        print("Pre-computation cancelled.")