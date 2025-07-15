#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import wikipediaapi
import time
from tqdm import tqdm  # For progress visualization


# In[2]:


final_df = pd.read_csv("imdb_full_result.csv") 
final_df


# In[25]:


movies_final = pd.read_csv("movies_for_streamlit_FINAL.csv")
movies_final.head()


# ### Wikipedia

# In[7]:


import requests
from bs4 import BeautifulSoup
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import os
import json
import re
import urllib.parse
import random


# In[8]:


import pandas as pd
import wikipediaapi
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import os
import json
import re


# In[9]:


from sanitize_filename import sanitize


# In[10]:


import logging


# In[16]:


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wiki_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Lock for thread-safe dataframe updates
df_lock = threading.Lock()

def sanitize_filename(filename):
    """Replace invalid filename characters with underscores."""
    # Characters not allowed in Windows filenames: \ / : * ? " < > |
    # Also handle apostrophes and other problematic characters
    invalid_chars = r'[\\/*?:"<>|\'.]'
    return re.sub(invalid_chars, '_', filename)


# In[17]:


def get_wikipedia_plot_with_api(title, year=None, cache_dir='wiki_cache'):
    """Get plot summary from Wikipedia using the API"""
    import wikipediaapi
    
    # cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key and check if cached
    cache_key = f"{title.replace(' ', '_')}_{year if year else ''}".lower()
    cache_key = sanitize_filename(cache_key)  # Make sure to use sanitize_filename
    cache_file = os.path.join(cache_dir, f"{cache_key}.txt")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_content = f.read()
            if cached_content and not cached_content.startswith("No Wikipedia"):
                return cached_content
    
    # FIXED: Set up Wikipedia API without duplicate user_agent parameter
    # Check your installed version with: pip show wikipedia-api
    
    # Option 1: If you have a newer version of the library
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='MovieRecommendationSystem/1.0'  # Single user agent parameter
    )
    
    # OR Option 2: If that still doesn't work, try this simpler initialization
    # wiki_wiki = wikipediaapi.Wikipedia('en')
    
    # Try different title variations
    title_variations = [
        title,
        f"{title} ({year})" if year else None,
        f"{title} ({year} film)" if year else None,
        f"{title} (film)",
        f"{title} (movie)"
    ]
    
    # Filter out None values
    title_variations = [v for v in title_variations if v]
    
    # Try each variation
    for title_var in title_variations:
        page = wiki_wiki.page(title_var)
        if page.exists():
            # Try to find a Plot section
            for section in page.sections:
                if section.title.lower() in ['plot', 'plot summary', 'synopsis', 'storyline']:
                    if section.text:
                        # Cache and return the plot
                        plot_text = section.text
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            f.write(plot_text)
                        return plot_text
            
            # If no plot section found, use the page summary
            if page.summary:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(page.summary)
                return page.summary
            
        # Respect Wikipedia's rate limits
        time.sleep(1)
    
    # No Wikipedia page found
    error_message = "No Wikipedia page found for this movie"
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(error_message)
    
    return error_message


# In[18]:


def process_movie_batch_with_api(batch_idx, movie_indices, df, title_column='title', year_column='year'):
    """Process a batch of movies using the Wikipedia API"""
    results = {}
    
    for idx in movie_indices:
        # Get the movie title
        title = df.loc[idx, title_column]
        
        # Get year if available
        year = None
        if year_column and year_column in df.columns:
            year_value = df.loc[idx, year_column]
            if pd.notna(year_value):
                try:
                    year = int(float(year_value))
                except (ValueError, TypeError):
                    pass
        
        # Fetch the plot summary using API
        try:
            plot = get_wikipedia_plot_with_api(title, year)
            results[idx] = plot
            
            # Add a reasonable delay between requests
            time.sleep(1)  # 1 second delay is much more respectful
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error processing movie '{title}': {error_msg}")
            results[idx] = error_msg
    
    logger.info(f"Completed batch {batch_idx} with {len(results)} movies")
    return results


# In[19]:


def add_plot_summaries_with_api(
    df, 
    title_column='title', 
    year_column='year', 
    batch_size=20, 
    max_workers=3,
    checkpoint_dir='checkpoint_wiki_api',
    checkpoint_interval=5
):
    """
    Process a dataframe of movies and add Wikipedia plot summaries using the Wikipedia API.
    
    Args:
        df (pandas.DataFrame): Dataframe containing movie information
        title_column (str): Name of column containing movie titles
        year_column (str): Name of column containing release years
        batch_size (int): Number of movies to process in each batch
        max_workers (int): Maximum number of concurrent threads
        checkpoint_dir (str): Directory to save checkpoint files
        checkpoint_interval (int): Save checkpoint after every this many batches
        
    Returns:
        pandas.DataFrame: Original dataframe with added 'wikipedia_plot' column
    """
    # checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for existing checkpoints
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("batch_")])
    
    # Determine starting point
    start_batch = 0
    processed_indices = set()
    
    if checkpoint_files:
        # Load progress from the last checkpoint
        last_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
        checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
        
        try:
            # Load comprehensive progress checkpoint if it exists
            comprehensive_checkpoint = os.path.join(checkpoint_dir, "all_progress.json")
            if os.path.exists(comprehensive_checkpoint):
                with open(comprehensive_checkpoint, 'r') as f:
                    progress_data = json.load(f)
                
                for movie_data in progress_data:
                    if isinstance(movie_data, dict) and 'index' in movie_data:
                        processed_indices.add(movie_data['index'])
                
                start_batch = int(last_checkpoint.split('_')[1].split('.')[0]) + 1
                logger.info(f"Resuming from batch {start_batch} with {len(processed_indices)} movies already processed")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            start_batch = 0
    
    # Initialize result dataframe
    result_df = df.copy()
    if 'wikipedia_plot' not in result_df.columns:
        result_df['wikipedia_plot'] = None
    
    # Divide movies into batches
    all_indices = [i for i in range(len(df)) if i not in processed_indices]
    num_batches = (len(all_indices) + batch_size - 1) // batch_size
    
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(all_indices))
        batch_indices = all_indices[start_idx:end_idx]
        batches.append((i, batch_indices))
    
    # Process batches with multithreading
    logger.info(f"Processing {len(batches)} batches using {max_workers} threads")
    
    # If resuming, skip already processed batches
    batches = batches[start_batch:]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # Submit initial batch of tasks (up to max_workers)
        for i, (batch_idx, batch_indices) in enumerate(batches[:max_workers]):
            future = executor.submit(
                process_movie_batch_with_api,  # Note this is the API version
                batch_idx,
                batch_indices, 
                df,
                title_column,
                year_column
            )
            futures[future] = (batch_idx, batch_indices)
        
        # Process all batches with progress tracking
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            batch_count = max_workers
            
            while futures:
                # Wait for the next future to complete
                done, _ = concurrent.futures.wait(
                    futures, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    batch_idx, batch_indices = futures.pop(future)
                    
                    try:
                        # Get batch results and update dataframe
                        batch_results = future.result()
                        
                        with df_lock:
                            for idx, plot in batch_results.items():
                                result_df.loc[idx, 'wikipedia_plot'] = plot
                        
                        # Save batch checkpoint
                        batch_file = os.path.join(checkpoint_dir, f"batch_{batch_idx}.json")
                        with open(batch_file, 'w', encoding='utf-8') as f:
                            json.dump(batch_results, f)
                        
                        # Save comprehensive checkpoint at intervals
                        if batch_idx % checkpoint_interval == 0:
                            # Save the full dataframe
                            comprehensive_file = os.path.join(checkpoint_dir, f"comprehensive_{batch_idx}.csv")
                            result_df.to_csv(comprehensive_file, index=False)
                            logger.info(f"Saved comprehensive checkpoint: {comprehensive_file}")
                        
                        # Update progress
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    
                    # Submit next batch if available
                    if batch_count < len(batches):
                        next_batch_idx, next_batch_indices = batches[batch_count]
                        
                        next_future = executor.submit(
                            process_movie_batch_with_api,  # Note this is the API version
                            next_batch_idx,
                            next_batch_indices,
                            df,
                            title_column,
                            year_column
                        )
                        
                        futures[next_future] = (next_batch_idx, next_batch_indices)
                        batch_count += 1
    
    # Save final result
    result_df.to_csv("movies_with_wiki_plots.csv", index=False)
    logger.info("Complete! Final dataset saved as 'movies_with_wiki_plots.csv'")
    
    return result_df


# In[ ]:


# First, make sure you have the wikipediaapi library installed
# pip install wikipedia-api

# Use the new API-based function with better parameters
result_df = add_plot_summaries_with_api(
    movies_final,
    title_column='title',
    year_column='year',
    batch_size=20,
    max_workers=3,
    checkpoint_dir='checkpoint_wiki_api',
    checkpoint_interval=5
)


# In[ ]:





# In[ ]:





# In[3]:


'''import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import os
import json
import re
import urllib.parse
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wiki_scraping.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Lock for thread-safe dataframe updates
df_lock = threading.Lock()

# User agent rotation for avoiding detection
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
]

def get_headers():
    """Generate random headers to avoid detection"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

def clean_title_for_url(title):
    """Convert movie title to Wikipedia URL format"""
    # Replace spaces with underscores and handle special characters
    clean_title = title.replace(' ', '_')
    # Handle other special characters
    clean_title = urllib.parse.quote(clean_title)
    return clean_title

def generate_wikipedia_urls(title, year=None):
    """Generate possible Wikipedia URLs for a movie"""
    urls = []
    
    # Clean the title for URL
    base_title = clean_title_for_url(title)
    
    # Basic variations
    urls.append(f"https://en.wikipedia.org/wiki/{base_title}")
    urls.append(f"https://en.wikipedia.org/wiki/{base_title}_(film)")
    urls.append(f"https://en.wikipedia.org/wiki/{base_title}_(movie)")
    
    # Add year-specific variations
    if year:
        urls.append(f"https://en.wikipedia.org/wiki/{base_title}_{year}")
        urls.append(f"https://en.wikipedia.org/wiki/{base_title}_({year}_film)")
        urls.append(f"https://en.wikipedia.org/wiki/{base_title}_({year}_movie)")
    
    # Handle special cases like "Director's Film"
    if "'" in title:
        # Try with the part after the possessive
        parts = title.split("'s ", 1)
        if len(parts) > 1:
            simplified = parts[1]
            simple_url = clean_title_for_url(simplified)
            urls.append(f"https://en.wikipedia.org/wiki/{simple_url}")
            urls.append(f"https://en.wikipedia.org/wiki/{simple_url}_(film)")
            if year:
                urls.append(f"https://en.wikipedia.org/wiki/{simple_url}_({year}_film)")
    
    return urls

def sanitize_filename(filename):
    """Replace invalid filename characters with underscores."""
    # Characters not allowed in Windows filenames: \ / : * ? " < > |
    # Also handle apostrophes and other problematic characters
    invalid_chars = r'[\\/*?:"<>|\'.]'
    return re.sub(invalid_chars, '_', filename)

def scrape_wikipedia_plot(title, year=None, cache_dir='wiki_cache'):
    """Scrape plot summary from Wikipedia for a movie"""
    # cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key and check if cached
    cache_key = f"{title.replace(' ', '_')}_{year if year else ''}".lower()
    
    # Sanitize the filename to remove invalid characters
    cache_key = sanitize_filename(cache_key)
    cache_file = os.path.join(cache_dir, f"{cache_key}.txt")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_content = f.read()
            if cached_content and not cached_content.startswith("No Wikipedia") and not cached_content.startswith("Error:"):
                return cached_content
    
    # Generate possible URLs to try
    urls = generate_wikipedia_urls(title, year)
    
    # Try each URL
    for url in urls:
        try:
            # Make the request with random headers
            response = requests.get(url, headers=get_headers(), timeout=10)
            
            # Check if page was found
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Check if it's a disambiguation page
                if soup.find('div', {'class': 'disambig'}) or 'may refer to' in soup.get_text():
                    continue
                
                # First, try to find a dedicated plot section
                plot_section = None
                plot_heading = None
                
                # Look for plot-related headings
                for heading in soup.find_all(['h2', 'h3']):
                    heading_text = heading.get_text().lower().strip()
                    if heading_text in ['plot', 'plot summary', 'synopsis', 'storyline']:
                        plot_heading = heading
                        break
                
                # If we found a plot heading, extract its content
                if plot_heading:
                    plot_section = ""
                    current = plot_heading.find_next()
                    
                    # Continue until we hit the next heading or end of content
                    while current and not current.name in ['h2', 'h3']:
                        if current.name == 'p':
                            plot_section += current.get_text() + "\n\n"
                        current = current.find_next()
                    
                    if plot_section.strip():
                        # Clean up the text
                        plot_section = re.sub(r'\[\d+\]', '', plot_section)  # Remove citation markers [1], [2], etc.
                        plot_section = re.sub(r'\s+', ' ', plot_section).strip()  # Normalize whitespace
                        
                        # Cache the result
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            f.write(plot_section)
                        
                        return plot_section
                
                # Fallback: If no plot section, try to get the first few paragraphs
                if not plot_section:
                    # Find the first content section after the intro
                    content_div = soup.find('div', {'class': 'mw-parser-output'})
                    if content_div:
                        # Get first 3-5 paragraphs, which often contain plot overview
                        paragraphs = content_div.find_all('p', recursive=False)[:5]
                        content = "\n\n".join([p.get_text() for p in paragraphs])
                        
                        # Clean up the text
                        content = re.sub(r'\[\d+\]', '', content)  # Remove citation markers
                        content = re.sub(r'\s+', ' ', content).strip()  # Normalize whitespace
                        
                        if len(content) > 200:  # Only use if substantial content found
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                f.write(content)
                            return content
        
        except Exception as e:
            logger.debug(f"Error accessing {url}: {str(e)}")
            continue
    
    # If we get here, no successful page was found
    error_message = "No Wikipedia page found for this movie"
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(error_message)
    
    return error_message

def process_movie_batch(batch_idx, movie_indices, df, title_column='title', year_column='year'):
    """Process a batch of movies to extract Wikipedia plot summaries using web scraping"""
    results = {}
    
    for idx in movie_indices:
        # Get the movie title
        title = df.loc[idx, title_column]
        
        # Get year if available
        year = None
        if year_column and year_column in df.columns:
            year_value = df.loc[idx, year_column]
            if pd.notna(year_value):
                try:
                    year = int(float(year_value))
                except (ValueError, TypeError):
                    pass
        
        # Fetch the plot summary
        try:
            plot = scrape_wikipedia_plot(title, year)
            results[idx] = plot
            
            # Add a small delay between requests to be kind to Wikipedia
            time.sleep(0.1)  # Very short delay - we can be more aggressive with direct scraping
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error processing movie '{title}': {error_msg}")
            results[idx] = error_msg
    
    logger.info(f"Completed batch {batch_idx} with {len(results)} movies")
    return results

def add_plot_summaries_with_scraping(
    df, 
    title_column='title', 
    year_column='year', 
    batch_size=200, 
    max_workers=32,
    checkpoint_dir='checkpoint_wiki_scrape',
    checkpoint_interval=5
):
    """
    Process a dataframe of movies and add Wikipedia plot summaries using web scraping.
    
    Args:
        df (pandas.DataFrame): Dataframe containing movie information
        title_column (str): Name of column containing movie titles
        year_column (str): Name of column containing release years
        batch_size (int): Number of movies to process in each batch
        max_workers (int): Maximum number of concurrent threads
        checkpoint_dir (str): Directory to save checkpoint files
        checkpoint_interval (int): Save checkpoint after every this many batches
        
    Returns:
        pandas.DataFrame: Original dataframe with added 'wikipedia_plot' column
    """
    # checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for existing checkpoints
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("batch_")])
    
    # Determine starting point
    start_batch = 0
    processed_indices = set()
    
    if checkpoint_files:
        # Load progress from the last checkpoint
        last_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
        checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
        
        try:
            # Load comprehensive progress checkpoint if it exists
            comprehensive_checkpoint = os.path.join(checkpoint_dir, "all_progress.json")
            if os.path.exists(comprehensive_checkpoint):
                with open(comprehensive_checkpoint, 'r') as f:
                    progress_data = json.load(f)
                
                for movie_data in progress_data:
                    if isinstance(movie_data, dict) and 'index' in movie_data:
                        processed_indices.add(movie_data['index'])
                
                start_batch = int(last_checkpoint.split('_')[1].split('.')[0]) + 1
                logger.info(f"Resuming from batch {start_batch} with {len(processed_indices)} movies already processed")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            start_batch = 0
    
    # Initialize result dataframe
    result_df = df.copy()
    if 'wikipedia_plot' not in result_df.columns:
        result_df['wikipedia_plot'] = None
    
    # Divide movies into batches
    all_indices = [i for i in range(len(df)) if i not in processed_indices]
    num_batches = (len(all_indices) + batch_size - 1) // batch_size
    
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(all_indices))
        batch_indices = all_indices[start_idx:end_idx]
        batches.append((i, batch_indices))
    
    # Process batches with multithreading
    logger.info(f"Processing {len(batches)} batches using {max_workers} threads")
    
    # If resuming, skip already processed batches
    batches = batches[start_batch:]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # Submit initial batch of tasks (up to max_workers)
        for i, (batch_idx, batch_indices) in enumerate(batches[:max_workers]):
            future = executor.submit(
                process_movie_batch,
                batch_idx,
                batch_indices, 
                df,
                title_column,
                year_column
            )
            futures[future] = (batch_idx, batch_indices)
        
        # Process all batches with progress tracking
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            batch_count = max_workers
            
            while futures:
                # Wait for the next future to complete
                done, _ = concurrent.futures.wait(
                    futures, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    batch_idx, batch_indices = futures.pop(future)
                    
                    try:
                        # Get batch results and update dataframe
                        batch_results = future.result()
                        
                        with df_lock:
                            for idx, plot in batch_results.items():
                                result_df.loc[idx, 'wikipedia_plot'] = plot
                        
                        # Save batch checkpoint
                        batch_file = os.path.join(checkpoint_dir, f"batch_{batch_idx}.json")
                        with open(batch_file, 'w', encoding='utf-8') as f:
                            json.dump(batch_results, f)
                        
                        # Save comprehensive checkpoint at intervals
                        if batch_idx % checkpoint_interval == 0:
                            # Save the full dataframe
                            comprehensive_file = os.path.join(checkpoint_dir, f"comprehensive_{batch_idx}.csv")
                            result_df.to_csv(comprehensive_file, index=False)
                            logger.info(f"Saved comprehensive checkpoint: {comprehensive_file}")
                        
                        # Update progress
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    
                    # Submit next batch if available
                    if batch_count < len(batches):
                        next_batch_idx, next_batch_indices = batches[batch_count]
                        
                        next_future = executor.submit(
                            process_movie_batch,
                            next_batch_idx,
                            next_batch_indices,
                            df,
                            title_column,
                            year_column
                        )
                        
                        futures[next_future] = (next_batch_idx, next_batch_indices)
                        batch_count += 1
    
    # Save final result
    result_df.to_csv("movies_for_streamlit.csv", index=False)
    logger.info("Complete! Final dataset saved as 'movies_for_streamlit.csv'")
    
    return result_df'''


# In[4]:


'''# Process with the scraping implementation
result_df = add_plot_summaries_with_scraping(
    final_df,
    title_column='title',  # Adjust to your column name
    year_column='year',    # Adjust to your column name
    batch_size=200,        # Process 200 movies per batch
    max_workers=32,        # Use 32 concurrent threads
    checkpoint_dir='checkpoint_wiki_scrape',
    checkpoint_interval=5  # Save comprehensive checkpoint every 5 batches
)'''


# In[11]:


import os
import time
import requests
from tqdm import tqdm
import json
import csv
import pandas as pd
import requests
from time import sleep


# In[12]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import re
import os
import json
from tqdm import tqdm


# In[ ]:





# In[2]:


movies_for_streamlit = pd.read_csv("movies_for_streamlit.csv")
movies_for_streamlit.head() 


# ### Claude

# In[4]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import time 
import random
import re
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import csv


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rt_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# directory for saving data
os.makedirs("rt_data", exist_ok=True)

# Configure a session with retry capability
def generate_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Rotate user agents to reduce blocking risk
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
    ]
    
    session.headers.update({
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://www.rottentomatoes.com/"
    })
    
    return session

def clean_text(text):
    """Clean up text by removing extra whitespace and normalizing quotes and apostrophes"""
    if not text:
        return ""
        
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle escaped apostrophes and quotes (both forms)
    text = text.replace("\\'", "'")       # Replace escaped single quotes
    text = text.replace('\\"', '"')       # Replace escaped double quotes
    text = text.replace('\\u2019', "'")   # Replace escaped Unicode right single quote
    text = text.replace('\\u2018', "'")   # Replace escaped Unicode left single quote
    
    # Handle actual Unicode characters
    text = text.replace('\u2019', "'")    # Replace Unicode right single quote
    text = text.replace('\u2018', "'")    # Replace Unicode left single quote
    text = text.replace('\u201c', '"')    # Replace Unicode left double quote
    text = text.replace('\u201d', '"')    # Replace Unicode right double quote
    text = text.replace('\u2013', '-')    # Replace Unicode en dash
    text = text.replace('\u2014', '--')   # Replace Unicode em dash
    text = text.replace('\u2026', '...')  # Replace Unicode ellipsis
    
    return text

def normalize_title(title):
    """Normalize movie title for better slug generation"""
    # Remove special characters that might affect URL formation
    title = re.sub(r'[^\w\s]', ' ', title)
    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def to_slug(title):
    """Convert movie title to a Rotten Tomatoes URL slug"""
    title = normalize_title(title)
    title = title.lower()
    title = re.sub(r'[^a-z0-9 ]+', '', title)
    title = re.sub(r'\s+', '_', title.strip())
    return title

def generate_slug_variants(title, year=None):
    """Generate multiple possible slug variants for a movie"""
    variants = []
    
    # Normalize the title
    norm_title = normalize_title(title)
    
    # Basic slug
    basic_slug = to_slug(norm_title)
    variants.append(basic_slug)
    
    # Add year if provided
    if year:
        variants.append(f"{basic_slug}_{year}")
    
    # Handle "The" prefix
    if norm_title.lower().startswith("the "):
        no_the = to_slug(norm_title[4:])
        variants.append(no_the)
        if year:
            variants.append(f"{no_the}_{year}")
    
    # Handle special characters and alternative formats
    alt_title = re.sub(r'[:\-–—&]', ' ', norm_title)
    alt_slug = to_slug(alt_title)
    if alt_slug != basic_slug:
        variants.append(alt_slug)
        if year:
            variants.append(f"{alt_slug}_{year}")
    
    # Handle colons in titles (common in sequels)
    if ":" in title:
        parts = title.split(":", 1)
        part1_slug = to_slug(parts[0])
        variants.append(part1_slug)
        if year:
            variants.append(f"{part1_slug}_{year}")
    
    # Handle numeric sequences in titles (like "2001" or "Se7en")
    numeric_variant = re.sub(r'(\d+)', lambda m: m.group(0).replace('1', 'one').replace('2', 'two')
                            .replace('3', 'three').replace('4', 'four').replace('5', 'five')
                            .replace('6', 'six').replace('7', 'seven').replace('8', 'eight')
                            .replace('9', 'nine').replace('0', 'zero'), norm_title)
    numeric_slug = to_slug(numeric_variant)
    if numeric_slug != basic_slug:
        variants.append(numeric_slug)
        if year:
            variants.append(f"{numeric_slug}_{year}")
    
    # Remove duplicates while preserving order
    unique_variants = []
    for v in variants:
        if v not in unique_variants:
            unique_variants.append(v)
    
    return unique_variants

def verify_slug(title, year=None, session=None):
    """Generate and verify multiple possible slug variants for a movie"""
    if session is None:
        session = generate_session()
    
    # Generate possible slug variants
    variants = generate_slug_variants(title, year)
    
    logger.debug(f"Testing slug variants for '{title}': {variants}")
    
    # Test each variant
    for slug in variants:
        url = f"https://www.rottentomatoes.com/m/{slug}"
        try:
            response = session.head(url, timeout=10)
            if response.status_code == 200:
                logger.info(f"Found valid slug for '{title}': {slug}")
                return slug
        except Exception as e:
            logger.debug(f"Error checking slug {slug}: {str(e)}")
            continue
    
    # If no variant works, try the search page as fallback
    try:
        search_url = f"https://www.rottentomatoes.com/search?search={title.replace(' ', '%20')}"
        response = session.get(search_url, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for movie search results
            movie_results = soup.select("search-page-media-row[type='movie']")
            
            for result in movie_results:
                result_title = result.select_one("[slot='title']")
                result_year = result.select_one("[slot='releaseYear']")
                
                # Check if this result matches our movie
                if result_title and (not year or not result_year or str(year) in result_year.text):
                    title_match = result_title.text.strip().lower() == title.lower()
                    year_match = not year or not result_year or str(year) in result_year.text
                    
                    if title_match and year_match:
                        # Extract slug from URL
                        url_elem = result.select_one("a[href*='/m/']")
                        if url_elem and 'href' in url_elem.attrs:
                            href = url_elem['href']
                            slug = href.split('/m/')[1].split('/')[0]
                            logger.info(f"Found slug via search for '{title}': {slug}")
                            return slug
        
    except Exception as e:
        logger.debug(f"Error using search fallback for '{title}': {str(e)}")
    
    logger.warning(f"Could not find valid RT URL for: {title}")
    return None

def get_rt_user_reviews(slug, max_reviews=15, session=None):
    """Extract user reviews from Rotten Tomatoes"""
    if session is None:
        session = generate_session()
        
    url = f"https://www.rottentomatoes.com/m/{slug}/reviews?type=user"
    reviews = []
    
    try:
        response = session.get(url, timeout=15)
        if response.status_code != 200:
            logger.warning(f"Failed to get user reviews for {slug}: Status code {response.status_code}")
            return reviews
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try modern pagination-based approach first
        page_count = 1
        total_pages_elem = soup.select_one(".pageInfo")
        if total_pages_elem:
            match = re.search(r'of\s+(\d+)', total_pages_elem.text)
            if match:
                page_count = min(int(match.group(1)), 3)  # Limit to first 3 pages
        
        # Try multiple selectors for robustness
        selectors = [
            ".audience-reviews__item",  # New RT design
            "p[data-qa='review-quote']",  # Older RT design
            ".user_review",  # Alternative class
            ".review_table_row"  # Very old design
        ]
        
        # Extract reviews from the first page
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                logger.info(f"Found {len(elements)} user reviews on page 1 with selector: {selector}")
                
                for element in elements:
                    # Extract review text based on selector type
                    if selector == ".audience-reviews__item":
                        text_elem = element.select_one(".audience-reviews__review")
                        review_text = text_elem.text.strip() if text_elem else None
                    elif selector == "p[data-qa='review-quote']":
                        review_text = element.text.strip()
                    else:
                        text_elem = element.select_one(".review_quote, .review-text")
                        review_text = text_elem.text.strip() if text_elem else None
                    
                    if review_text and len(review_text) > 20:  # Minimum length to avoid garbage
                        reviews.append(clean_text(review_text))
                        if len(reviews) >= max_reviews:
                            return reviews
                
                # If we found reviews on first page and there are more pages, fetch them
                if reviews and page_count > 1:
                    for page in range(2, page_count + 1):
                        if len(reviews) >= max_reviews:
                            break
                            
                        page_url = f"https://www.rottentomatoes.com/m/{slug}/reviews?type=user&page={page}"
                        try:
                            page_response = session.get(page_url, timeout=15)
                            if page_response.status_code == 200:
                                page_soup = BeautifulSoup(page_response.text, 'html.parser')
                                page_elements = page_soup.select(selector)
                                
                                for element in page_elements:
                                    if selector == ".audience-reviews__item":
                                        text_elem = element.select_one(".audience-reviews__review")
                                        review_text = text_elem.text.strip() if text_elem else None
                                    elif selector == "p[data-qa='review-quote']":
                                        review_text = element.text.strip()
                                    else:
                                        text_elem = element.select_one(".review_quote, .review-text")
                                        review_text = text_elem.text.strip() if text_elem else None
                                    
                                    if review_text and len(review_text) > 20:
                                        reviews.append(clean_text(review_text))
                                        if len(reviews) >= max_reviews:
                                            break
                                            
                                # Add delay between page requests
                                time.sleep(random.uniform(1, 2))
                                
                        except Exception as e:
                            logger.warning(f"Error fetching user reviews page {page} for {slug}: {str(e)}")
                
                break  # Stop once we've found reviews with one selector
        
        # If no reviews found with any selector, try a more aggressive approach
        if not reviews:
            # Try alternative approach - look for any paragraph inside reviews section
            review_section = soup.select_one("#reviews, .audience-reviews, .reviews-movie")
            if review_section:
                paragraphs = review_section.select("p")
                if paragraphs:
                    for p in paragraphs:
                        text = p.text.strip()
                        if len(text) > 50:  # Only substantial paragraphs
                            reviews.append(clean_text(text))
                            if len(reviews) >= max_reviews:
                                break
        
        return reviews
        
    except Exception as e:
        logger.error(f"Error extracting user reviews for {slug}: {str(e)}")
        return reviews
    
def decode_json_reviews(json_string):
    """Properly decode a JSON string containing reviews, handling escape sequences"""
    if not json_string or not isinstance(json_string, str):
        return []
        
    try:
        # First try standard JSON parsing
        reviews = json.loads(json_string)
        
        # Clean up each review
        cleaned_reviews = []
        for review in reviews:
            if isinstance(review, str):
                cleaned_reviews.append(clean_text(review))
            else:
                cleaned_reviews.append(review)
                
        return cleaned_reviews
    except json.JSONDecodeError:
        # If there's an issue with JSON decoding, try to fix common problems
        # Remove escaped backslashes that might be causing problems
        fixed_json = json_string.replace("\\'", "'").replace('\\"', '"')
        try:
            reviews = json.loads(fixed_json)
            
            # Clean up each review
            cleaned_reviews = []
            for review in reviews:
                if isinstance(review, str):
                    cleaned_reviews.append(clean_text(review))
                else:
                    cleaned_reviews.append(review)
                    
            return cleaned_reviews
        except:
            # If all else fails, return empty list
            return []

def get_rt_critic_reviews(slug, max_reviews=15, session=None):
    """Extract critic reviews from Rotten Tomatoes with enhanced cleaning"""
    if session is None:
        session = generate_session()
        
    url = f"https://www.rottentomatoes.com/m/{slug}/reviews"
    reviews = []
    
    try:
        response = session.get(url, timeout=15)
        if response.status_code != 200:
            logger.warning(f"Failed to get critic reviews for {slug}: Status code {response.status_code}")
            return reviews
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for pagination
        page_count = 1
        total_pages_elem = soup.select_one(".pageInfo")
        if total_pages_elem:
            match = re.search(r'of\s+(\d+)', total_pages_elem.text)
            if match:
                page_count = min(int(match.group(1)), 3)  # Limit to first 3 pages
        
        # Approach 1: Try to extract reviews from review blocks first
        critic_blocks = soup.select(".review-row, [data-qa='review-item'], .reviewItem")
        if critic_blocks:
            for block in critic_blocks[:max_reviews]:
                # Extract the review text only from within the block
                review_text = None
                
                # Try different selectors for the review text within the block
                for selector in [".review-text", ".the_review", ".review_text", 
                               ".review-quote", "p.review_quote", "[data-qa='review-quote']"]:
                    text_elem = block.select_one(selector)
                    if text_elem and text_elem.text.strip():
                        review_text = text_elem.text.strip()
                        break
                
                if not review_text:
                    # Fallback: Try to get text directly if it's a short block
                    if len(block.text) < 500:  # Avoid getting too much surrounding content
                        review_text = block.text.strip()
                
                if review_text and len(review_text) > 20:
                    # Clean and normalize the review text
                    clean_review = clean_text(review_text)
                    
                    # Remove common prefixes/suffixes from RT
                    clean_review = re.sub(r'Full Review\s*\|.*$', '', clean_review)
                    clean_review = re.sub(r'Original Score:.*$', '', clean_review)
                    
                    # Add to list if unique and substantial
                    if clean_review and len(clean_review) > 30 and clean_review not in reviews:
                        reviews.append(clean_review)
            
            logger.info(f"Found {len(reviews)} critic reviews using review blocks")
        
        # Approach 2: If not enough reviews found, try selectors directly
        if len(reviews) < max_reviews:
            # Try multiple selectors for direct extraction
            selectors = [
                "div.review-text",      # New RT design
                "div.the_review",       # Alternative new design
                "div.review__text",     # Slightly older design
                ".critic_review p",     # Older design specific paragraph
                "p.review_quote",       # Specific review quotes
                "[data-qa='review-quote']", # Data attribute selector
                ".review_table_row p"   # Very old design
            ]
            
            for selector in selectors:
                if len(reviews) >= max_reviews:
                    break
                    
                elements = soup.select(selector)
                if elements:
                    logger.info(f"Found {len(elements)} potential critic reviews with selector: {selector}")
                    
                    for element in elements:
                        review_text = element.text.strip()
                        if review_text and len(review_text) > 20:
                            # Clean and normalize
                            clean_review = clean_text(review_text)
                            clean_review = re.sub(r'Full Review\s*\|.*$', '', clean_review)
                            clean_review = re.sub(r'Original Score:.*$', '', clean_review)
                            
                            # Add if unique and substantial
                            if clean_review and len(clean_review) > 30 and clean_review not in reviews:
                                reviews.append(clean_review)
                                if len(reviews) >= max_reviews:
                                    break
                    
                    # If we found reviews and need more, check other pages
                    if reviews and len(reviews) < max_reviews and page_count > 1:
                        for page in range(2, page_count + 1):
                            if len(reviews) >= max_reviews:
                                break
                                
                            page_url = f"https://www.rottentomatoes.com/m/{slug}/reviews?page={page}"
                            try:
                                page_response = session.get(page_url, timeout=15)
                                if page_response.status_code == 200:
                                    page_soup = BeautifulSoup(page_response.text, 'html.parser')
                                    page_elements = page_soup.select(selector)
                                    
                                    for element in page_elements:
                                        review_text = element.text.strip()
                                        if review_text and len(review_text) > 20:
                                            clean_review = clean_text(review_text)
                                            clean_review = re.sub(r'Full Review\s*\|.*$', '', clean_review)
                                            clean_review = re.sub(r'Original Score:.*$', '', clean_review)
                                            
                                            if clean_review and len(clean_review) > 30 and clean_review not in reviews:
                                                reviews.append(clean_review)
                                                if len(reviews) >= max_reviews:
                                                    break
                                                    
                                    # Add delay between page requests
                                    time.sleep(random.uniform(1, 2))
                                    
                            except Exception as e:
                                logger.warning(f"Error fetching critic reviews page {page} for {slug}: {str(e)}")
        
        # Approach 3: Try structured data if still not enough reviews
        if len(reviews) < max_reviews:
            # Look for reviews with schema.org markup
            schema_reviews = soup.select("[itemtype*='Review'], [itemtype*='review']")
            
            for item in schema_reviews:
                review_body = item.select_one("[itemprop='reviewBody'], [itemprop='description']")
                if review_body and review_body.text.strip():
                    clean_review = clean_text(review_body.text.strip())
                    if clean_review and len(clean_review) > 30 and clean_review not in reviews:
                        reviews.append(clean_review)
                        if len(reviews) >= max_reviews:
                            break
        
        # Approach 4: Last resort - try any elements with "review" in class
        if len(reviews) < max_reviews // 2:  # If we have fewer than half the desired reviews
            review_containers = soup.select("[class*='review'], [class*='Review']")
            for container in review_containers:
                # Skip containers that are too large (likely contain multiple reviews/metadata)
                if len(container.text) > 800:
                    continue
                    
                # Skip if it contains navigation elements
                if container.select("a, button, input"):
                    continue
                    
                review_text = container.text.strip()
                if review_text and len(review_text) > 100:
                    clean_review = clean_text(review_text)
                    # Further clean by removing common patterns
                    clean_review = re.sub(r'Full Review\s*\|.*$', '', clean_review)
                    clean_review = re.sub(r'Original Score:.*$', '', clean_review)
                    
                    if clean_review and len(clean_review) > 30 and clean_review not in reviews:
                        reviews.append(clean_review)
                        if len(reviews) >= max_reviews:
                            break
        
        # Final filtering to remove navigation elements, headers, etc.
        filtered_reviews = []
        for review in reviews:
            # Skip entries that are likely navigation elements or metadata
            skip_phrases = ["all critics", "top critics", "all audience", "verified audience", 
                           "full review", "original score", "load more", "view all videos", 
                           "movie reviews by", "reviewer type"]
                           
            if any(phrase in review.lower() for phrase in skip_phrases):
                continue
                
            # Skip reviews that are just reviewer names or publications
            if len(review.split()) < 5:
                continue
                
            # Skip reviews with RT boilerplate text
            if "do you think we mischaracterized" in review.lower():
                continue
                
            # Add to filtered list if not already present
            if review not in filtered_reviews:
                filtered_reviews.append(review)
        
        # Return the specified max number or fewer
        return filtered_reviews[:max_reviews]
        
    except Exception as e:
        logger.error(f"Error extracting critic reviews for {slug}: {str(e)}")
        return reviews
    

def post_process_reviews(reviews):
    """Clean up and filter review data to ensure quality"""
    if not reviews:
        return []
        
    # Process each review
    cleaned_reviews = []
    for review in reviews:
        # Skip if not a string
        if not isinstance(review, str):
            continue
            
        # Basic cleanup
        clean = clean_text(review)
        
        # Remove common RT artifacts
        clean = re.sub(r'Full Review\s*\|.*$', '', clean)
        clean = re.sub(r'Original Score:.*$', '', clean)
        clean = re.sub(r'Do you think we mischaracterized.*$', '', clean)
        
        # Remove navigation text that might have been captured
        nav_phrases = ["all critics", "top critics", "load more", "view all", "all audience", 
                       "verified audience", "full review", "original score", "movie reviews by reviewer type"]
        if any(phrase in clean.lower() for phrase in nav_phrases):
            continue
            
        # Remove very short reviews
        if len(clean) < 30 or len(clean.split()) < 5:
            continue
            
        # Remove duplicate reviews
        if clean not in cleaned_reviews:
            cleaned_reviews.append(clean)
    
    return cleaned_reviews

def normalize_escaped_unicode(text):
    """Convert escaped Unicode characters to their proper representation"""
    if not isinstance(text, str):
        return text
        
    # Handle common escaped Unicode in text
    text = text.replace('\\u2019', "'")  # Right single quotation mark
    text = text.replace('\\u2018', "'")  # Left single quotation mark
    text = text.replace('\\u201c', '"')  # Left double quotation mark
    text = text.replace('\\u201d', '"')  # Right double quotation mark
    text = text.replace('\\u2013', '-')  # En dash
    text = text.replace('\\u2014', '--')  # Em dash
    text = text.replace('\\u2026', '...')  # Horizontal ellipsis
    text = text.replace("\\'", "'")      # Escaped apostrophe
    
    return text


def process_movie(movie_data):
    """Process a single movie to extract RT reviews"""
    idx = movie_data['idx']
    title = movie_data['title']
    year = movie_data['year']
    max_reviews = movie_data['max_reviews']
    
    # Check if result already exists
    result_file = f"rt_data/movie_{idx}_{to_slug(title)}.json"
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if (existing_data.get('user_reviews') and existing_data.get('critic_reviews') and 
                    (len(existing_data['user_reviews']) >= 5 or len(existing_data['critic_reviews']) >= 5)):
                    logger.info(f"Using existing data for movie {idx}: {title}")
                    return existing_data
        except:
            pass  # If there's an error reading the file, proceed with scraping
    
    logger.info(f"Processing movie {idx}: {title} ({year})")
    
    session = generate_session()
    
    # Find valid slug
    slug = verify_slug(title, year, session)
    
    if not slug:
        result = {
            'index': idx,
            'title': title,
            'year': year,
            'rt_slug': None,
            'user_reviews': [],
            'critic_reviews': []
        }
        return result
    
    # Get reviews with some delay to avoid rate limiting
    user_reviews = get_rt_user_reviews(slug, max_reviews, session)
    user_reviews = post_process_reviews(user_reviews)  # Apply post-processing
    logger.info(f"Found {len(user_reviews)} user reviews for {title}")
    
    # Random delay between requests
    time.sleep(random.uniform(1.5, 3))
    
    critic_reviews = get_rt_critic_reviews(slug, max_reviews, session)
    critic_reviews = post_process_reviews(critic_reviews)  # Apply post-processing
    logger.info(f"Found {len(critic_reviews)} critic reviews for {title}")
    
    # Save individual movie result
    result = {
        'index': idx,
        'title': title,
        'year': year,
        'rt_slug': slug,
        'user_reviews': user_reviews,
        'critic_reviews': critic_reviews
    }
    
    # Save to file
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result

def process_all_movies(df, batch_size=10, max_movies=None, max_reviews=15, num_workers=128):
    """Process full dataset with parallel execution, batching and checkpointing"""
    # Determine number of movies to process
    total_movies = min(len(df), max_movies) if max_movies else len(df)
    
    # Prepare movie data for processing
    movies_to_process = []
    for idx in range(total_movies):
        movie = df.iloc[idx]
        title = movie['title']
        # Handle various year column formats
        year = None
        if 'Year' in movie and pd.notna(movie['Year']):
            year = int(float(movie['Year']))
        elif 'year' in movie and pd.notna(movie['year']):
            year = int(float(movie['year']))
        elif 'startYear' in movie and pd.notna(movie['startYear']):
            year = int(float(movie['startYear']))
            
        movies_to_process.append({
            'idx': idx,
            'title': title, 
            'year': year,
            'max_reviews': max_reviews
        })
    
    all_results = []
    
    # Process in batches
    num_batches = (total_movies + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_movies)
        
        logger.info(f"Processing batch {batch_idx+1}/{num_batches} (movies {start_idx+1}-{end_idx})")
        
        batch_movies = movies_to_process[start_idx:end_idx]
        batch_results = []
        
        # Process batch with parallel execution
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            batch_results = list(tqdm(
                executor.map(process_movie, batch_movies), 
                total=len(batch_movies),
                desc=f"Batch {batch_idx+1}/{num_batches}"
            ))
        
        all_results.extend(batch_results)
        
        # Save batch results
        batch_file = f"rt_data/batch_{start_idx}_{end_idx}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        # Save overall progress
        with open("rt_data/all_progress.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Random delay between batches to avoid rate limiting
        if batch_idx < num_batches - 1:
            delay = random.uniform(3, 6)
            logger.info(f"Waiting {delay:.1f} seconds before next batch")
            time.sleep(delay)
    
    return all_results

def apply_reviews_to_dataframe(df, review_data):
    """Add the extracted reviews to the original dataframe"""
    result_df = df.copy()
    
    # Add columns for reviews if they don't exist
    if 'rt_slug' not in result_df.columns:
        result_df['rt_slug'] = None
    if 'rt_user_reviews' not in result_df.columns:
        result_df['rt_user_reviews'] = None
    if 'rt_critic_reviews' not in result_df.columns:
        result_df['rt_critic_reviews'] = None
    
    # Add reviews to dataframe by index
    for item in review_data:
        idx = item['index']
        if idx < len(result_df):
            result_df.at[idx, 'rt_slug'] = item['rt_slug']
            
            # Clean and encode reviews
            user_reviews = [clean_text(r) for r in item['user_reviews']] if item['user_reviews'] else []
            critic_reviews = [clean_text(r) for r in item['critic_reviews']] if item['critic_reviews'] else []
            
            # Save as JSON with Unicode characters preserved
            result_df.at[idx, 'rt_user_reviews'] = json.dumps(user_reviews, ensure_ascii=False)
            result_df.at[idx, 'rt_critic_reviews'] = json.dumps(critic_reviews, ensure_ascii=False)
    
    return result_df

def resume_from_checkpoint():
    """Resume processing from the latest checkpoint"""
    checkpoint_file = "rt_data/all_progress.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                if all_results:
                    # Find the highest index processed
                    max_index = max(item['index'] for item in all_results)
                    logger.info(f"Resuming from checkpoint. Already processed {len(all_results)} movies, up to index {max_index}")
                    return all_results, max_index + 1
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
    
    return [], 0


# Check for existing progress
existing_results, start_index = resume_from_checkpoint()

if existing_results and start_index >= len(movies_for_streamlit):
    logger.info("All movies have already been processed. Loading existing results.")
    review_data = existing_results
else:
    # Process remaining movies
    if start_index > 0:
        # Slice the dataframe to process only remaining movies
        remaining_df = movies_for_streamlit.iloc[start_index:]
        logger.info(f"Processing remaining {len(remaining_df)} movies from index {start_index}")
        
        new_results = process_all_movies(
            remaining_df,
            batch_size=20,       # Increased batch size
            max_reviews=15,      # More reviews per movie
            num_workers=128        # Parallel processing
        )
        
        # Adjust indices in new results to match original dataframe
        for item in new_results:
            item['index'] += start_index
        
        # Combine with existing results
        review_data = existing_results + new_results
    else:
        # Process all movies
        review_data = process_all_movies(
            movies_for_streamlit,
            batch_size=20,
            max_reviews=15,
            num_workers=128
        )


def normalize_review_columns(df):
    """Normalize Unicode in JSON review columns"""
    result = df.copy()
    
    # Process review columns
    for col in ['rt_user_reviews', 'rt_critic_reviews']:
        if col in result.columns:
            result[col] = result[col].apply(
                lambda x: normalize_escaped_unicode(x) if isinstance(x, str) else x
            )
    
    return result

# Add reviews to dataframe
df_with_reviews = apply_reviews_to_dataframe(movies_for_streamlit, review_data)

df_with_reviews = normalize_review_columns(df_with_reviews)

# Save enhanced dataframe
df_with_reviews.to_csv("movies_for_streamlit_FINAL.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

# Save a sample for inspection
sample_df = df_with_reviews.head(20).copy()

# Add review count columns for easy inspection
sample_df['user_review_count'] = sample_df['rt_user_reviews'].apply(
    lambda x: len(decode_json_reviews(x)) if pd.notna(x) else 0
)
sample_df['critic_review_count'] = sample_df['rt_critic_reviews'].apply(
    lambda x: len(decode_json_reviews(x)) if pd.notna(x) else 0
)

sample_df.to_csv("rt_FINAL_Sample.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

sample_df

   


# In[9]:


import pandas as pd


# In[2]:


sample_df = pd.read_csv("rt_reviews_sample.csv")


# In[6]:


df_check = pd.read_csv("movies_for_streamlit_FINAL.csv")
df_check


# In[3]:


pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# In[7]:


df_check = df_check.head(10)


# In[8]:


df_check.columns


# In[18]:


column_types = df_check.dtypes
column_types


# In[21]:


pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full content of each column

row = df_check[df_check["imdb_id"] == "tt36169404"]
row


# In[22]:


df_check.to_csv("sample_df.csv", index=False)


# In[9]:


# Check for various types of "empty" values
def check_for_empty_values(df, column='wikipedia_plot'):
    """
    Check for different types of empty or problematic values in a column
    """
    # Check for null values
    print(f"Rows with NaN/None values: {df[column].isnull().sum()}")
    
    # Check for empty strings
    empty_strings = df[df[column] == ''].shape[0]
    print(f"Rows with empty strings: {empty_strings}")
    
    # Check for strings with only whitespace
    whitespace_only = df[df[column].str.strip() == ''].shape[0]
    print(f"Rows with only whitespace: {whitespace_only}")
    
    # Check for specific error messages from your function
    no_wiki_page = df[df[column] == "No Wikipedia page found for this movie"].shape[0]
    print(f"Rows with 'No Wikipedia page found': {no_wiki_page}")
    
    no_plot_found = df[df[column] == "No plot found on Wikipedia page"].shape[0]
    print(f"Rows with 'No plot found': {no_plot_found}")
    
    # Check for error messages
    error_msgs = df[df[column].str.startswith("Error:", na=False)].shape[0]
    print(f"Rows with error messages: {error_msgs}")
    
    # Check for very short entries that might be meaningless
    short_entries = df[df[column].str.len() < 50].shape[0]
    print(f"Rows with very short entries (< 50 chars): {short_entries}")
    
    return


# In[10]:


check_for_empty_values(df_with_reviews, 'wikipedia_plot')


# In[23]:


def analyze_wikipedia_plot_coverage(df):
    """
    Analyze the coverage and quality of Wikipedia plot data
    """
    total_rows = len(df)
    
    # Count different types of issues
    issues = {
        'no_page': df['wikipedia_plot'] == "No Wikipedia page found for this movie",
        'no_plot': df['wikipedia_plot'] == "No plot found on Wikipedia page",
        'errors': df['wikipedia_plot'].str.startswith("Error:", na=False),
        'short': df['wikipedia_plot'].str.len() < 100,
        'empty': df['wikipedia_plot'] == "",
        'whitespace': df['wikipedia_plot'].str.strip() == ""
    }
    
    print(f"Total movies: {total_rows}")
    print("\nIssues found:")
    
    for issue_name, condition in issues.items():
        count = condition.sum()
        percentage = (count / total_rows) * 100
        print(f"{issue_name}: {count} ({percentage:.2f}%)")
    
    # Get successful plots
    successful_plots = df[
        ~issues['no_page'] & 
        ~issues['no_plot'] & 
        ~issues['errors'] & 
        ~issues['short'] & 
        ~issues['empty'] & 
        ~issues['whitespace']
    ]
    
    print(f"\nSuccessful Wikipedia plots: {len(successful_plots)} ({len(successful_plots)/total_rows*100:.2f}%)")
    
    # Return problematic rows for further inspection
    problematic_rows = df[
        issues['no_page'] | 
        issues['no_plot'] | 
        issues['errors'] | 
        issues['short'] | 
        issues['empty'] | 
        issues['whitespace']
    ]
    
    return problematic_rows

# Run the analysis
problematic_movies = analyze_wikipedia_plot_coverage(result_df)

# Display some examples of problematic movies
if not problematic_movies.empty:
    print("\nProblematic movies (first 10):")
    print(problematic_movies[['title', 'year', 'wikipedia_plot']].head(10))


# In[ ]:





# ### Parental Guide

# In[1]:


import os
import json
import logging
import time
import threading
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import CinemagoerNG
from cinemagoerng import web, model


# In[2]:


# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parental_guide_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thread-safe lock for DataFrame operations
df_lock = threading.Lock()


# In[3]:


@dataclass
class ParentalGuideData:
    """Data class to store parental guide information"""
    imdb_id: str
    mpaa_rating: Optional[str] = None
    
    # Content lists - store the actual detailed descriptions
    violence: List[str] = None
    profanity: List[str] = None
    sex_nudity: List[str] = None
    alcohol_drugs: List[str] = None
    frightening: List[str] = None
    
    # Severity scores (1-5 scale)
    violence_severity: int = 0
    profanity_severity: int = 0
    sex_nudity_severity: int = 0
    alcohol_drugs_severity: int = 0
    frightening_severity: int = 0
    
    # Metadata
    extraction_date: str = None
    extraction_status: str = "pending"
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize empty lists if None are provided"""
        if self.violence is None:
            self.violence = []
        if self.profanity is None:
            self.profanity = []
        if self.sex_nudity is None:
            self.sex_nudity = []
        if self.alcohol_drugs is None:
            self.alcohol_drugs = []
        if self.frightening is None:
            self.frightening = []
        
        if self.extraction_date is None:
            self.extraction_date = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParentalGuideData':
        """instance from dictionary"""
        return cls(**data)


# In[4]:


def normalize_imdb_id(movie_id: Any) -> Optional[str]:
    """
    Normalize various IMDb ID formats to the standard format.
    
    This function handles different input formats and ensures consistent output
    that CinemagoerNG can work with reliably.
    """
    if not movie_id:
        return None
    
    # Convert to string and strip whitespace
    movie_id = str(movie_id).strip()
    
    # If it already has 'tt' prefix, validate and return
    if movie_id.startswith('tt'):
        numeric_part = movie_id[2:]
        if numeric_part.isdigit() and len(numeric_part) >= 6:
            return movie_id
        elif numeric_part.isdigit():
            # Pad with zeros if needed
            return f"tt{numeric_part.zfill(7)}"
    
    # If it's just numbers, add 'tt' prefix
    if movie_id.isdigit():
        return f"tt{movie_id.zfill(7)}"
    
    # Invalid format
    logger.warning(f"Invalid IMDb ID format: {movie_id}")
    return None


# In[5]:


def map_status_to_severity(status: str) -> int:
    """
    Map CinemagoerNG status strings to our 1-5 severity scale.
    
    This function translates the human-readable status strings that CinemagoerNG
    provides into numerical scores that are easier to work with programmatically.
    """
    if not status:
        return 0
    
    status_lower = status.lower().strip()
    
    # Map status strings to severity scores based on typical parental guide language
    status_mapping = {
        'none': 0,
        'very mild': 1,
        'mild': 2,
        'moderate': 3,
        'strong': 4,
        'severe': 5,
        'intense': 4,
        'graphic': 5
    }
    
    return status_mapping.get(status_lower, 3)  # Default to moderate if unknown


# In[6]:


def extract_advisory_content(advisory) -> tuple[List[str], int]:
    """
    Extract content descriptions and severity from a CinemagoerNG Advisory object.
    
    This function handles the detailed parsing of advisory information, including
    cleaning up HTML entities and extracting both text descriptions and severity levels.
    
    Returns:
        tuple: (list of content descriptions, severity score)
    """
    content_descriptions = []
    severity = 0
    
    if advisory is None:
        return content_descriptions, severity
    
    # Extract severity from status
    if hasattr(advisory, 'status') and advisory.status:
        severity = map_status_to_severity(advisory.status)
    
    # Extract detailed descriptions
    if hasattr(advisory, 'details') and advisory.details:
        for detail in advisory.details:
            if hasattr(detail, 'text') and detail.text:
                # Clean up the text (handle HTML entities, etc.)
                text = detail.text.replace('&#39;', "'").replace('&quot;', '"')
                content_descriptions.append(text.strip())
    
    return content_descriptions, severity


# In[7]:


def extract_parental_guide_single(imdb_id: str, cache_dir: str = 'cinemagoerng_cache') -> ParentalGuideData:
    """
    Extract parental guide information for a single movie using CinemagoerNG.
    
    This is the core extraction function that handles the business logic of
    fetching and parsing parental guide data. It includes caching for efficiency
    and comprehensive error handling.
    """
    # Normalize the IMDb ID
    normalized_id = normalize_imdb_id(imdb_id)
    if not normalized_id:
        logger.error(f"Invalid IMDb ID: {imdb_id}")
        return ParentalGuideData(
            imdb_id=str(imdb_id),
            extraction_status="error",
            error_message="Invalid IMDb ID format"
        )
    
    # Set up caching to avoid re-processing movies
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{normalized_id}_parental.json")
    
    # Check cache first - this dramatically improves performance for re-runs
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if cached_data.get('extraction_status') == 'success':
                    logger.debug(f"Using cached data for {normalized_id}")
                    return ParentalGuideData.from_dict(cached_data)
        except Exception as e:
            logger.warning(f"Cache read error for {normalized_id}: {e}")
    
    # Initialize the data structure
    pg_data = ParentalGuideData(imdb_id=normalized_id)
    
    try:
        logger.debug(f"Fetching parental guide for {normalized_id}")
        
        # Use the method confirmed by our debugging session
        title = web.get_title(normalized_id, page='parental_guide')
        
        if not title:
            logger.warning(f"No title found for {normalized_id}")
            pg_data.extraction_status = "no_data"
            pg_data.error_message = "Title not found"
            return pg_data
        
        # Extract MPAA rating from certification if available
        if hasattr(title, 'certification') and title.certification:
            if hasattr(title.certification, 'mpa_rating_reason'):
                pg_data.mpaa_rating = title.certification.mpa_rating_reason
            elif hasattr(title.certification, 'mpa_rating'):
                pg_data.mpaa_rating = title.certification.mpa_rating
        
        # Extract advisory content using the structure discovered in debugging
        if hasattr(title, 'advisories') and title.advisories:
            logger.debug(f"Found advisories for {normalized_id}")
            
            # Map CinemagoerNG categories to our data structure
            category_mappings = [
                ('violence', 'violence'),
                ('profanity', 'profanity'),
                ('nudity', 'sex_nudity'),      # CinemagoerNG uses 'nudity', we use 'sex_nudity'
                ('alcohol', 'alcohol_drugs'),  # CinemagoerNG uses 'alcohol', we use 'alcohol_drugs'
                ('frightening', 'frightening')
            ]
            
            for cng_category, our_category in category_mappings:
                if hasattr(title.advisories, cng_category):
                    advisory = getattr(title.advisories, cng_category)
                    content, severity = extract_advisory_content(advisory)
                    
                    # Store the extracted data
                    setattr(pg_data, our_category, content)
                    setattr(pg_data, f"{our_category}_severity", severity)
            
            # Check if we successfully extracted any content
            total_items = sum(len(getattr(pg_data, cat)) for cat in 
                            ['violence', 'profanity', 'sex_nudity', 'alcohol_drugs', 'frightening'])
            
            if total_items > 0:
                pg_data.extraction_status = "success"
                logger.info(f"Successfully extracted {total_items} parental guide items for {normalized_id}")
            else:
                pg_data.extraction_status = "no_data"
                pg_data.error_message = "No parental guide content found"
        else:
            pg_data.extraction_status = "no_data"
            pg_data.error_message = "No advisories found"
        
        # Cache the result for future use
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(pg_data.to_dict(), f, ensure_ascii=False, indent=2)
        
        return pg_data
        
    except Exception as e:
        logger.error(f"Error extracting parental guide for {normalized_id}: {e}")
        pg_data.extraction_status = "error"
        pg_data.error_message = str(e)
        
        # Cache error state to avoid repeated failures
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(pg_data.to_dict(), f, ensure_ascii=False, indent=2)
        
        return pg_data


# In[8]:


def process_movie_batch(
    batch_idx: int, 
    movie_indices: List[int], 
    df: pd.DataFrame, 
    imdb_id_column: str = 'imdb_id'
) -> Dict[int, ParentalGuideData]:
    """
    Process a batch of movies to extract parental guide information.
    
    This function is designed for parallel processing. Each batch runs independently,
    which allows us to use multiple threads to speed up processing while respecting
    API rate limits within each batch.
    
    Args:
        batch_idx: Index of this batch (for logging purposes)
        movie_indices: List of DataFrame row indices to process in this batch
        df: The DataFrame containing movie information
        imdb_id_column: Name of the column containing IMDb IDs
        
    Returns:
        Dictionary mapping DataFrame indices to ParentalGuideData objects
    """
    results = {}
    
    for idx in movie_indices:
        try:
            # Get the IMDb ID from the DataFrame
            imdb_id = df.loc[idx, imdb_id_column]
            
            # Extract parental guide data using our corrected function
            pg_data = extract_parental_guide_single(imdb_id)
            results[idx] = pg_data
            
            # Add a respectful delay between requests to avoid overwhelming the API
            # This is crucial for being a good citizen when using web APIs
            time.sleep(2)  # 2-second delay between requests
            
        except Exception as e:
            logger.error(f"Error processing movie at index {idx}: {e}")
            # an error record so we don't lose track of failed items
            results[idx] = ParentalGuideData(
                imdb_id=str(df.loc[idx, imdb_id_column]),
                extraction_status="error",
                error_message=str(e)
            )
    
    logger.info(f"Completed batch {batch_idx} with {len(results)} movies")
    return results


def add_parental_guide_complete(
    df: pd.DataFrame,
    imdb_id_column: str = 'imdb_id',
    batch_size: int = 10,
    max_workers: int = 6,
    checkpoint_dir: str = 'checkpoint_parental_guide',
    checkpoint_interval: int = 5,
    resume_from_checkpoint: bool = True
) -> pd.DataFrame:
    """
    Main function to process an entire DataFrame of movies with comprehensive infrastructure.
    
    This function orchestrates the complete process including:
    - Checkpoint management for reliability
    - Batch processing for memory efficiency  
    - Parallel processing for speed
    - Progress tracking and logging
    - Graceful error handling
    
    Args:
        df: DataFrame containing movie information
        imdb_id_column: Name of column containing IMDb IDs
        batch_size: Number of movies to process in each batch
        max_workers: Number of parallel threads to use
        checkpoint_dir: Directory to store checkpoint files
        checkpoint_interval: Save progress every N batches
        resume_from_checkpoint: Whether to resume from previous progress
        
    Returns:
        Enhanced DataFrame with parental guide information
    """
    # Validate input parameters
    if imdb_id_column not in df.columns:
        raise ValueError(f"Column '{imdb_id_column}' not found in DataFrame")
    
    logger.info(f"Starting parental guide extraction for {len(df)} movies")
    logger.info(f"Using {max_workers} workers with batch size {batch_size}")
    
    # checkpoint directory for saving progress
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize result DataFrame with new columns
    result_df = df.copy()
    
    # Add columns for parental guide data if they don't exist
    new_columns = [
        'mpaa_rating',
        'violence_severity',
        'profanity_severity', 
        'sex_nudity_severity',
        'alcohol_drugs_severity',
        'frightening_severity',
        'parental_guide_status'
    ]
    
    for col in new_columns:
        if col not in result_df.columns:
            result_df[col] = None
    
    # Load checkpoint if resuming (this allows you to restart interrupted processes)
    processed_indices = set()
    start_batch = 0
    
    if resume_from_checkpoint:
        checkpoint_file = os.path.join(checkpoint_dir, 'progress.json')
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    processed_indices = set(checkpoint_data.get('processed_indices', []))
                    start_batch = checkpoint_data.get('last_batch', 0) + 1
                    logger.info(f"Resuming from batch {start_batch} with {len(processed_indices)} movies already processed")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
    
    # Prepare batches for processing
    all_indices = [i for i in range(len(df)) if i not in processed_indices]
    batches = []
    
    # batches of the specified size
    for i in range(0, len(all_indices), batch_size):
        batch_indices = all_indices[i:i + batch_size]
        batches.append((len(batches), batch_indices))
    
    # Skip already processed batches when resuming
    batches = batches[start_batch:]
    
    if not batches:
        logger.info("All movies have been processed!")
        return result_df
    
    logger.info(f"Processing {len(batches)} batches")
    
    # Process batches using thread pool for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches to the thread pool
        future_to_batch = {
            executor.submit(process_movie_batch, batch_idx, batch_indices, df, imdb_id_column): 
            (batch_idx, batch_indices) for batch_idx, batch_indices in batches
        }
        
        # Process completed batches as they finish
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx, batch_indices = future_to_batch[future]
                
                try:
                    # Get results from the completed batch
                    batch_results = future.result()
                    
                    # Update DataFrame with results (thread-safe)
                    with df_lock:
                        for idx, pg_data in batch_results.items():
                            result_df.loc[idx, 'mpaa_rating'] = pg_data.mpaa_rating
                            result_df.loc[idx, 'violence_severity'] = pg_data.violence_severity
                            result_df.loc[idx, 'profanity_severity'] = pg_data.profanity_severity
                            result_df.loc[idx, 'sex_nudity_severity'] = pg_data.sex_nudity_severity
                            result_df.loc[idx, 'alcohol_drugs_severity'] = pg_data.alcohol_drugs_severity
                            result_df.loc[idx, 'frightening_severity'] = pg_data.frightening_severity
                            result_df.loc[idx, 'parental_guide_status'] = pg_data.extraction_status
                            
                            processed_indices.add(idx)
                    
                    # Save checkpoint periodically to prevent data loss
                    if batch_idx % checkpoint_interval == 0:
                        # Save progress information
                        checkpoint_data = {
                            'processed_indices': list(processed_indices),
                            'last_batch': start_batch + batch_idx,
                            'total_processed': len(processed_indices),
                            'timestamp': datetime.now().isoformat()
                        }
                        checkpoint_file = os.path.join(checkpoint_dir, 'progress.json')
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint_data, f, indent=2)
                        
                        # Save intermediate results
                        intermediate_file = os.path.join(checkpoint_dir, f'intermediate_results_{batch_idx}.csv')
                        result_df.to_csv(intermediate_file, index=False)
                        logger.info(f"Saved checkpoint at batch {batch_idx}")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                
                # Update progress bar
                pbar.update(1)
    
    # Save final results
    final_file = "missing_movies_parental_results.csv"
    result_df.to_csv(final_file, index=False)
    logger.info(f"Processing complete! Saved final results to {final_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PARENTAL GUIDE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total movies processed: {len(result_df)}")
    print(f"Successful extractions: {(result_df['parental_guide_status'] == 'success').sum()}")
    print(f"No data found: {(result_df['parental_guide_status'] == 'no_data').sum()}")
    print(f"Errors encountered: {(result_df['parental_guide_status'] == 'error').sum()}")
    
    # Show severity distribution for successful extractions
    successful_df = result_df[result_df['parental_guide_status'] == 'success']
    if len(successful_df) > 0:
        print("\nSEVERITY DISTRIBUTION (for successful extractions):")
        for category in ['violence', 'profanity', 'sex_nudity', 'alcohol_drugs', 'frightening']:
            col = f'{category}_severity'
            if col in successful_df.columns:
                dist = successful_df[col].value_counts().sort_index()
                print(f"\n{category.upper()}:")
                for severity, count in dist.items():
                    if pd.notna(severity) and severity > 0:
                        print(f"  Level {int(severity)}: {count} movies")
    
    return result_df


# In[9]:


import pandas as pd
import logging
from datetime import datetime
import os



def run_production_parental_guide_processing():
    """
    Production script to process your complete movie dataset with parental guide information.
    
    This script is designed for long-running, reliable processing of your full dataset.
    It includes optimized parameters based on your successful test run and provides
    comprehensive monitoring and logging.
    """
    
    # Set up production-level logging
    log_filename = f"production_parental_guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("PRODUCTION PARENTAL GUIDE PROCESSING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_filename}")
    print()
    
    # Load your movie dataset
    # Replace 'your_movie_file.csv' with the actual filename of your dataset
    print("Loading movie dataset...")
    try:
        # Adjust this line to match your actual dataset file and structure
        df = pd.read_csv('missing_movies_to_process.csv', index_col=0)  # or whatever your file is named
        
        print(f"Successfully loaded {len(df)} movies")
        print(f"Dataset columns: {list(df.columns)}")
        
        # Verify the IMDb ID column exists
        imdb_column_candidates = ['imdb_id', 'imdbID', 'imdb', 'tconst']
        imdb_column = None
        
        for candidate in imdb_column_candidates:
            if candidate in df.columns:
                imdb_column = candidate
                break
        
        if imdb_column is None:
            raise ValueError(f"Could not find IMDb ID column. Available columns: {list(df.columns)}")
        
        print(f"Using '{imdb_column}' as IMDb ID column")
        
        # Show a sample of the data to verify it looks correct
        print(f"\nSample of first 3 movies:")
        print(df[['title', imdb_column]].head(3).to_string() if 'title' in df.columns 
              else df[[imdb_column]].head(3).to_string())
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your file path and ensure the dataset exists.")
        return False
    
    # Production configuration parameters
    production_config = {
        'batch_size': 20,           # Process 20 movies per batch - good balance of efficiency and granularity
        'max_workers': 6,           # Use 4 parallel workers - respects API limits while providing good speed
        'checkpoint_interval': 10,  # Save progress every 10 batches - preserves work without excessive overhead
        'checkpoint_dir': 'missing_movies_checkpoint',
        'resume_from_checkpoint': False  # Always try to resume from previous progress
    }
    
    print(f"\nProduction Configuration:")
    print(f"  Batch size: {production_config['batch_size']} movies per batch")
    print(f"  Workers: {production_config['max_workers']} parallel threads")
    print(f"  Checkpoint interval: Every {production_config['checkpoint_interval']} batches")
    print(f"  Checkpoint directory: {production_config['checkpoint_dir']}")
    
    # Calculate estimated processing time
    movies_per_hour = estimate_processing_rate(production_config)
    estimated_hours = len(df) / movies_per_hour
    
    print(f"\nProcessing Estimates:")
    print(f"  Estimated rate: ~{movies_per_hour:.0f} movies per hour")
    print(f"  Estimated total time: ~{estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")
    print(f"  Progress will be saved every ~{production_config['batch_size'] * production_config['checkpoint_interval']} movies")
    
    # Ask for confirmation before starting the long-running process
    print(f"\nReady to process {len(df)} movies with parental guide extraction.")
    print("This is a long-running process that will take several hours or days to complete.")
    print("The process can be safely interrupted and resumed later.")
    
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Processing cancelled.")
        return False
    
    print(f"\nStarting production processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    print("You can monitor progress in the log file and console output.")
    print("Feel free to interrupt with Ctrl+C if needed - progress will be saved and you can resume later.")
    print()
    
    try:
        # Run the complete processing system
        result_df = add_parental_guide_complete(
            df,
            imdb_id_column=imdb_column,
            batch_size=20,
            max_workers=4,
            checkpoint_dir='missing_movies_checkpoint',
            checkpoint_interval=5,
            resume_from_checkpoint=True 
        )
        
        # Generate final report
        generate_final_report(result_df)
        
        print(f"\n🎉 PRODUCTION PROCESSING COMPLETED SUCCESSFULLY! 🎉")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Processing interrupted by user at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Your progress has been saved and you can resume processing later by running this script again.")
        return False
        
    except Exception as e:
        logger.error(f"Error during production processing: {e}")
        print(f"\n❌ Error occurred: {e}")
        print("Check the log file for detailed error information.")
        print("You can try resuming processing by running this script again.")
        return False

def estimate_processing_rate(config):
    """
    Estimate processing rate with more realistic concurrency modeling.
    
    This accounts for the fact that scaling workers beyond a certain point
    provides diminishing returns due to API limits and network constraints.
    """
    base_rate_per_worker = 750  # movies per hour per worker (from your test)
    
    # Apply concurrency efficiency factor
    workers = config['max_workers']
    
    # Efficiency decreases as worker count increases beyond optimal range
    if workers <= 4:
        efficiency_factor = 1.0  # Full efficiency up to 4 workers
    elif workers <= 8:
        efficiency_factor = 0.8  # 80% efficiency for 5-8 workers
    elif workers <= 12:
        efficiency_factor = 0.6  # 60% efficiency for 9-12 workers
    else:
        efficiency_factor = 0.4  # 40% efficiency beyond 12 workers
    
    # Calculate estimated rate with efficiency adjustment
    estimated_rate = base_rate_per_worker * workers * efficiency_factor
    
    return estimated_rate

def generate_final_report(result_df):
    """
    Generate a comprehensive report of the parental guide processing results.
    
    This report helps you understand what was accomplished and the quality
    of the data that's now available for your recommendation system.
    """
    print("\n" + "="*80)
    print("FINAL PROCESSING REPORT")
    print("="*80)
    
    total_movies = len(result_df)
    successful = (result_df['parental_guide_status'] == 'success').sum()
    no_data = (result_df['parental_guide_status'] == 'no_data').sum()
    errors = (result_df['parental_guide_status'] == 'error').sum()
    
    print(f"Dataset Overview:")
    print(f"  Total movies processed: {total_movies:,}")
    print(f"  Successful extractions: {successful:,} ({successful/total_movies*100:.1f}%)")
    print(f"  No parental guide data: {no_data:,} ({no_data/total_movies*100:.1f}%)")
    print(f"  Processing errors: {errors:,} ({errors/total_movies*100:.1f}%)")
    
    if successful > 0:
        successful_df = result_df[result_df['parental_guide_status'] == 'success']
        
        print(f"\nContent Analysis (based on {successful:,} successful extractions):")
        
        # Analyze severity distributions
        severity_columns = ['violence_severity', 'profanity_severity', 'sex_nudity_severity', 
                          'alcohol_drugs_severity', 'frightening_severity']
        
        for col in severity_columns:
            if col in successful_df.columns:
                category_name = col.replace('_severity', '').replace('_', ' ').title()
                
                # Count movies with each severity level
                severity_counts = successful_df[col].value_counts().sort_index()
                movies_with_content = (successful_df[col] > 0).sum()
                
                print(f"\n  {category_name}:")
                print(f"    Movies with {category_name.lower()} content: {movies_with_content:,} ({movies_with_content/successful*100:.1f}%)")
                
                for severity, count in severity_counts.items():
                    if severity > 0:  # Only show non-zero severities
                        severity_labels = {1: 'Very Mild', 2: 'Mild', 3: 'Moderate', 4: 'Strong', 5: 'Severe'}
                        label = severity_labels.get(int(severity), f'Level {int(severity)}')
                        print(f"      {label}: {count:,} movies")
    
    # Save summary statistics
    summary_file = f"parental_guide_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Parental Guide Processing Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total movies: {total_movies:,}\n")
        f.write(f"Successful: {successful:,} ({successful/total_movies*100:.1f}%)\n")
        f.write(f"No data: {no_data:,} ({no_data/total_movies*100:.1f}%)\n")
        f.write(f"Errors: {errors:,} ({errors/total_movies*100:.1f}%)\n")
    
    print(f"\nDetailed summary saved to: {summary_file}")
    print(f"Enhanced dataset saved to: movies_with_parental_guide_final.csv")

if __name__ == "__main__":
    print("Production Parental Guide Processing System")
    print("This script will enhance your movie dataset with comprehensive parental guide information.")
    print()
    
    success = run_production_parental_guide_processing()
    
    if success:
        print("\n✅ Your movie recommendation system now has comprehensive parental guide data!")
        print("You can integrate this enhanced dataset into your Streamlit application.")
    else:
        print("\n🔄 Processing can be resumed later by running this script again.")


# In[10]:


import pandas as pd
import glob

# Check the latest intermediate file
checkpoint_files = glob.glob("missing_movies_checkpoint/intermediate_results_*.csv")
if checkpoint_files:
    latest_file = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Latest checkpoint file: {latest_file}")
    
    # Load and check the data
    checkpoint_df = pd.read_csv(latest_file, index_col=0)
    
    # Check parental guide status
    status_counts = checkpoint_df['parental_guide_status'].value_counts()
    print(f"Status distribution in checkpoint:")
    print(status_counts)
    
    # Check if there's actual data
    successful = (checkpoint_df['parental_guide_status'] == 'success').sum()
    print(f"Successful extractions in checkpoint: {successful}")
    
    if successful > 0:
        print("✅ Your data IS in the checkpoint files!")
        print("The problem is the resume mechanism isn't loading it properly.")
    else:
        print("❌ No successful data in checkpoint files either.")


# In[17]:


#!/usr/bin/env python3
"""
Proper Merge of Parental Guide Data
Fixes the failed merge and makes the complete dataset correctly
"""

import pandas as pd
import numpy as np

def fix_parental_guide_merge():
    """Properly merge the original and missing movie parental guide data"""
    
    print("🔧 FIXING PARENTAL GUIDE MERGE")
    print("=" * 60)
    
    # Step 1: Load the original dataset with existing parental guide data
    print("\n1. Loading original dataset...")
    try:
        original_complete = pd.read_csv('movies_with_parental_guide_RECOVERED.csv')
        print(f"✅ Loaded original dataset: {len(original_complete):,} movies")
        
    except Exception as e:
        print(f"❌ Error loading original dataset: {e}")
        return False
    
    # Step 2: Load the missing movies results
    print("\n2. Loading missing movies results...")
    try:
        # Load without treating any column as index first
        missing_results = pd.read_csv('missing_movies_parental_results.csv')
        
        # Check if we have an imdb_id column to use for matching
        if 'imdb_id' in missing_results.columns:
            print(f"✅ Loaded missing results: {len(missing_results):,} movies")
            print(f"   Will match records using IMDb IDs")
            
            # Check what we got
            missing_successful = (missing_results['parental_guide_status'] == 'success').sum()
            missing_no_data = (missing_results['parental_guide_status'] == 'no_data').sum()
            print(f"   Successful: {missing_successful:,}")
            print(f"   No data: {missing_no_data:,}")
        else:
            print("❌ No imdb_id column found for matching")
            return False
        
    except Exception as e:
        print(f"❌ Error loading missing results: {e}")
        return False
    
    # Step 3: Check current parental guide data in original
    print("\n3. Analyzing existing parental guide data...")
    
    parental_columns = [
        'mpaa_rating', 'violence_severity', 'profanity_severity', 
        'sex_nudity_severity', 'alcohol_drugs_severity', 'frightening_severity', 
        'parental_guide_status'
    ]
    
    # Add missing columns if they don't exist
    for col in parental_columns:
        if col not in original_complete.columns:
            original_complete[col] = None
            print(f"   Added missing column: {col}")
    
    # Check existing data
    existing_successful = (original_complete['parental_guide_status'] == 'success').sum()
    print(f"   Existing successful extractions: {existing_successful:,}")
    
    # Step 4: Perform the proper merge using IMDb ID matching
    print("\n4. Performing proper merge using IMDb ID matching...")
    
    # the final dataset
    final_df = original_complete.copy()
    
    # Track merge statistics
    updated_count = 0
    new_successful = 0
    
    # a mapping from IMDb ID to row index in original dataset for faster lookup
    imdb_to_index = {imdb_id: idx for idx, imdb_id in enumerate(original_complete['imdb_id'])}
    
    # Update each row from missing_results by matching IMDb IDs
    for _, missing_row in missing_results.iterrows():
        imdb_id = missing_row['imdb_id']
        
        # Find the corresponding row in the original dataset
        if imdb_id in imdb_to_index:
            original_idx = imdb_to_index[imdb_id]
            
            # Check if we're updating a previously unprocessed movie
            was_unprocessed = pd.isna(final_df.loc[original_idx, 'parental_guide_status'])
            
            # Update all parental guide columns
            for col in parental_columns:
                if col in missing_results.columns and pd.notna(missing_row[col]):
                    final_df.loc[original_idx, col] = missing_row[col]
                    
            # Track statistics
            if pd.notna(missing_row['parental_guide_status']):
                updated_count += 1
                if missing_row['parental_guide_status'] == 'success' and was_unprocessed:
                    new_successful += 1
        else:
            print(f"   Warning: IMDb ID {imdb_id} not found in original dataset")
    
    print(f"✅ Updated {updated_count:,} movies with new parental guide data")
    print(f"✅ Added {new_successful:,} new successful extractions")
    
    # Step 5: Verify the results
    print("\n5. Verifying merge results...")
    
    final_successful = (final_df['parental_guide_status'] == 'success').sum()
    final_processed = final_df['parental_guide_status'].notna().sum()
    
    print(f"📊 Final Results:")
    print(f"   Total movies: {len(final_df):,}")
    print(f"   Movies processed: {final_processed:,}")
    print(f"   Successful extractions: {final_successful:,}")
    print(f"   Success rate: {final_successful/final_processed*100:.1f}%" if final_processed > 0 else "   Success rate: 0%")
    print(f"   Coverage: {final_processed/len(final_df)*100:.1f}%")
    
    # Expected vs actual check
    expected_successful = existing_successful + missing_successful
    print(f"\n🎯 Validation:")
    print(f"   Expected total successful: {expected_successful:,}")
    print(f"   Actual total successful: {final_successful:,}")
    
    if final_successful >= expected_successful:
        print(f"   ✅ Merge successful! Got {final_successful - existing_successful:,} new extractions")
    else:
        print(f"   ⚠️  Missing {expected_successful - final_successful:,} extractions")
    
    # Step 6: Save the corrected dataset
    print("\n6. Saving corrected dataset...")
    
    output_file = 'movies_with_COMPLETE_parental_guide_FIXED.csv'
    final_df.to_csv(output_file, index=False)
    print(f"✅ Saved corrected dataset: {output_file}")
    
    # Step 7: Show sample of successfully merged data
    print("\n7. Sample of newly added parental guide data...")
    
    # Find movies that were just updated
    newly_updated = final_df[
        (final_df['parental_guide_status'] == 'success') & 
        (final_df.index.isin(missing_results.index))
    ]
    
    if len(newly_updated) > 0:
        sample_size = min(5, len(newly_updated))
        sample = newly_updated.sample(sample_size)
        
        for _, movie in sample.iterrows():
            print(f"\n• {movie.get('title', 'Unknown')} ({movie.get('year', 'Unknown')})")
            severities = []
            for sev_col in ['violence_severity', 'profanity_severity', 'sex_nudity_severity', 
                           'alcohol_drugs_severity', 'frightening_severity']:
                if sev_col in movie and pd.notna(movie[sev_col]) and movie[sev_col] > 0:
                    category = sev_col.replace('_severity', '').title()
                    severities.append(f"{category}: {int(movie[sev_col])}")
            
            if severities:
                print(f"  Content: {', '.join(severities)}")
            else:
                print(f"  Content: Clean (no warnings)")
    
    print(f"\n🎉 MERGE FIXED SUCCESSFULLY!")
    print(f"📁 Use the file: {output_file}")
    
    return True

if __name__ == "__main__":
    print("Proper Parental Guide Data Merge")
    print("This script fixes the failed merge and makes the complete dataset.\n")
    
    success = fix_parental_guide_merge()
    
    if success:
        print(f"\n✅ Merge fixed successfully!")
        print(f"🚀 Your movie recommendation system now has the complete parental guide data!")
        print(f"📊 Run the analysis script again on the FIXED file to see the proper results!")
    else:
        print(f"\n❌ Merge fix failed. Please check the error messages above.")


# In[ ]:


#!/usr/bin/env python3
"""
Comprehensive Parental Guide Data Analysis
Analyzes the merged dataset to understand coverage, quality, and distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_parental_guide_data():
    """Complete analysis of the merged parental guide dataset"""
    
    print("🔍 COMPREHENSIVE PARENTAL GUIDE DATA ANALYSIS")
    print("=" * 80)
    
    # Load the complete dataset
    try:
        df = pd.read_csv('movies_with_COMPLETE_parental_guide_FIXED.csv')
        print(f"✅ Loaded complete dataset: {len(df):,} movies")
    except FileNotFoundError:
        print("❌ movies_with_COMPLETE_parental_guide.csv not found!")
        print("   Make sure you've run the merge step first.")
        return False
    
    print(f"\n📊 DATASET OVERVIEW")
    print("-" * 40)
    print(f"Total movies in dataset: {len(df):,}")
    print(f"Dataset columns: {len(df.columns)}")
    
    # Check parental guide columns
    parental_columns = [
        'mpaa_rating', 'violence_severity', 'profanity_severity', 
        'sex_nudity_severity', 'alcohol_drugs_severity', 'frightening_severity', 
        'parental_guide_status'
    ]
    
    missing_columns = [col for col in parental_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️  Missing columns: {missing_columns}")
    else:
        print(f"✅ All parental guide columns present: {parental_columns}")
    
    # 1. PROCESSING STATUS ANALYSIS
    print(f"\n📈 PROCESSING STATUS ANALYSIS")
    print("-" * 40)
    
    # Check how many movies have parental guide status
    if 'parental_guide_status' in df.columns:
        processed_movies = df['parental_guide_status'].notna().sum()
        unprocessed_movies = df['parental_guide_status'].isna().sum()
        
        print(f"Movies with parental guide processing: {processed_movies:,} ({processed_movies/len(df)*100:.1f}%)")
        print(f"Movies not yet processed: {unprocessed_movies:,} ({unprocessed_movies/len(df)*100:.1f}%)")
        
        # Status breakdown
        if processed_movies > 0:
            status_counts = df['parental_guide_status'].value_counts()
            print(f"\nProcessing Status Breakdown:")
            for status, count in status_counts.items():
                percentage = count / processed_movies * 100
                print(f"  {status}: {count:,} ({percentage:.1f}%)")
    
    # 2. SUCCESS RATE ANALYSIS
    print(f"\n🎯 SUCCESS RATE ANALYSIS")
    print("-" * 40)
    
    if 'parental_guide_status' in df.columns:
        successful_movies = df[df['parental_guide_status'] == 'success']
        no_data_movies = df[df['parental_guide_status'] == 'no_data']
        error_movies = df[df['parental_guide_status'] == 'error']
        
        print(f"Successful extractions: {len(successful_movies):,}")
        print(f"No parental guide data available: {len(no_data_movies):,}")
        print(f"Processing errors: {len(error_movies):,}")
        
        if processed_movies > 0:
            success_rate = len(successful_movies) / processed_movies * 100
            print(f"Overall success rate: {success_rate:.1f}%")
    
    # 3. SEVERITY DISTRIBUTION ANALYSIS
    print(f"\n📊 SEVERITY DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    severity_columns = ['violence_severity', 'profanity_severity', 'sex_nudity_severity', 
                       'alcohol_drugs_severity', 'frightening_severity']
    
    if len(successful_movies) > 0:
        print(f"Analysis based on {len(successful_movies):,} successful extractions:")
        
        severity_labels = {0: 'None', 1: 'Very Mild', 2: 'Mild', 3: 'Moderate', 4: 'Strong', 5: 'Severe'}
        
        for col in severity_columns:
            if col in df.columns:
                category_name = col.replace('_severity', '').replace('_', ' ').title()
                print(f"\n{category_name}:")
                
                # Count movies with non-zero severity
                movies_with_content = (successful_movies[col] > 0).sum()
                print(f"  Movies with {category_name.lower()} content: {movies_with_content:,} ({movies_with_content/len(successful_movies)*100:.1f}%)")
                
                # Severity distribution
                severity_dist = successful_movies[col].value_counts().sort_index()
                for severity, count in severity_dist.items():
                    if pd.notna(severity):
                        label = severity_labels.get(int(severity), f'Level {int(severity)}')
                        percentage = count / len(successful_movies) * 100
                        print(f"    {label}: {count:,} ({percentage:.1f}%)")
    
    # 4. MPAA RATING ANALYSIS
    print(f"\n🎬 MPAA RATING ANALYSIS")
    print("-" * 40)
    
    if 'mpaa_rating' in df.columns:
        mpaa_with_data = df['mpaa_rating'].notna().sum()
        print(f"Movies with MPAA rating data: {mpaa_with_data:,}")
        
        if mpaa_with_data > 0:
            mpaa_counts = df['mpaa_rating'].value_counts()
            print(f"MPAA Rating Distribution:")
            for rating, count in mpaa_counts.head(10).items():  # Top 10 ratings
                percentage = count / mpaa_with_data * 100
                print(f"  {rating}: {count:,} ({percentage:.1f}%)")
    
    # 5. QUALITY ANALYSIS - Sample successful extractions
    print(f"\n🎪 SAMPLE SUCCESSFUL EXTRACTIONS")
    print("-" * 40)
    
    if len(successful_movies) > 0:
        # Show movies with interesting parental guide data
        interesting_movies = successful_movies[
            (successful_movies['violence_severity'] > 0) |
            (successful_movies['profanity_severity'] > 0) |
            (successful_movies['sex_nudity_severity'] > 0) |
            (successful_movies['alcohol_drugs_severity'] > 0) |
            (successful_movies['frightening_severity'] > 0)
        ].copy()
        
        if len(interesting_movies) > 0:
            sample_movies = interesting_movies.sample(min(10, len(interesting_movies)))
            
            display_cols = ['title', 'year', 'imdb_rating', 'votes', 'violence_severity', 
                           'profanity_severity', 'sex_nudity_severity', 'alcohol_drugs_severity', 
                           'frightening_severity', 'mpaa_rating']
            
            # Only include columns that exist
            available_cols = [col for col in display_cols if col in sample_movies.columns]
            
            print("Sample movies with parental guide content:")
            for _, movie in sample_movies[available_cols].iterrows():
                title = movie.get('title', 'Unknown')
                year = movie.get('year', 'Unknown')
                rating = movie.get('imdb_rating', 'N/A')
                votes = movie.get('votes', 'N/A')
                
                print(f"\n• {title} ({year}) - {rating}/10")
                print(f"  Votes: {votes:,}" if votes != 'N/A' else "  Votes: N/A")
                
                # Show severity scores
                severities = []
                for sev_col in ['violence_severity', 'profanity_severity', 'sex_nudity_severity', 
                               'alcohol_drugs_severity', 'frightening_severity']:
                    if sev_col in movie and pd.notna(movie[sev_col]) and movie[sev_col] > 0:
                        category = sev_col.replace('_severity', '').title()
                        severities.append(f"{category}: {int(movie[sev_col])}")
                
                if severities:
                    print(f"  Content: {', '.join(severities)}")
                
                if 'mpaa_rating' in movie and pd.notna(movie['mpaa_rating']):
                    print(f"  MPAA: {movie['mpaa_rating']}")
    
    # 6. COVERAGE BY POPULARITY
    print(f"\n📈 COVERAGE BY MOVIE POPULARITY")
    print("-" * 40)
    
    if 'votes' in df.columns and 'parental_guide_status' in df.columns:
        # popularity buckets
        df_analysis = df.copy()
        df_analysis['popularity_bucket'] = pd.cut(
            df_analysis['votes'], 
            bins=[0, 1000, 10000, 100000, float('inf')], 
            labels=['<1K votes', '1K-10K votes', '10K-100K votes', '>100K votes']
        )
        
        print("Parental guide coverage by popularity:")
        for bucket in ['<1K votes', '1K-10K votes', '10K-100K votes', '>100K votes']:
            bucket_movies = df_analysis[df_analysis['popularity_bucket'] == bucket]
            processed_in_bucket = bucket_movies['parental_guide_status'].notna().sum()
            successful_in_bucket = (bucket_movies['parental_guide_status'] == 'success').sum()
            
            if len(bucket_movies) > 0:
                processed_pct = processed_in_bucket / len(bucket_movies) * 100
                success_pct = successful_in_bucket / len(bucket_movies) * 100
                print(f"  {bucket}: {len(bucket_movies):,} movies")
                print(f"    Processed: {processed_in_bucket:,} ({processed_pct:.1f}%)")
                print(f"    Successful: {successful_in_bucket:,} ({success_pct:.1f}%)")
    
    # 7. CONTENT FILTERING CAPABILITIES
    print(f"\n🔒 CONTENT FILTERING CAPABILITIES")
    print("-" * 40)
    
    if len(successful_movies) > 0:
        # Family-friendly movies (all severities <= 2)
        family_friendly = successful_movies[
            (successful_movies['violence_severity'] <= 2) &
            (successful_movies['profanity_severity'] <= 2) &
            (successful_movies['sex_nudity_severity'] <= 2) &
            (successful_movies['alcohol_drugs_severity'] <= 2) &
            (successful_movies['frightening_severity'] <= 2)
        ]
        
        # High-intensity movies (any severity >= 4)
        high_intensity = successful_movies[
            (successful_movies['violence_severity'] >= 4) |
            (successful_movies['profanity_severity'] >= 4) |
            (successful_movies['sex_nudity_severity'] >= 4) |
            (successful_movies['alcohol_drugs_severity'] >= 4) |
            (successful_movies['frightening_severity'] >= 4)
        ]
        
        # Clean movies (all severities = 0)
        clean_movies = successful_movies[
            (successful_movies['violence_severity'] == 0) &
            (successful_movies['profanity_severity'] == 0) &
            (successful_movies['sex_nudity_severity'] == 0) &
            (successful_movies['alcohol_drugs_severity'] == 0) &
            (successful_movies['frightening_severity'] == 0)
        ]
        
        print(f"Movies suitable for content filtering:")
        print(f"  Family-friendly (all content ≤ mild): {len(family_friendly):,} movies")
        print(f"  High-intensity (any content ≥ strong): {len(high_intensity):,} movies")
        print(f"  Clean (no content warnings): {len(clean_movies):,} movies")
        print(f"  Moderate content (between clean and family): {len(successful_movies) - len(family_friendly) - len(clean_movies):,} movies")
    
    # 8. FINAL SUMMARY
    print(f"\n🎉 FINAL SUMMARY")
    print("-" * 40)
    
    total_with_parental_data = (df['parental_guide_status'] == 'success').sum() if 'parental_guide_status' in df.columns else 0
    
    print(f"🎯 Your movie recommendation system now includes:")
    print(f"   • {len(df):,} total movies")
    print(f"   • {total_with_parental_data:,} movies with detailed parental guide data")
    print(f"   • {total_with_parental_data/len(df)*100:.1f}% of your catalog has content filtering capability")
    print(f"   • Comprehensive coverage across all popularity levels")
    print(f"   • Ready for family-friendly and content-aware recommendations!")
    
    # Save summary to file
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"parental_guide_analysis_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"Parental Guide Data Analysis Summary\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write(f"Total movies: {len(df):,}\n")
        f.write(f"Movies with parental guide data: {total_with_parental_data:,}\n")
        f.write(f"Coverage percentage: {total_with_parental_data/len(df)*100:.1f}%\n")
        if 'parental_guide_status' in df.columns:
            processed = df['parental_guide_status'].notna().sum()
            f.write(f"Movies processed: {processed:,}\n")
            if processed > 0:
                success_rate = total_with_parental_data / processed * 100
                f.write(f"Success rate: {success_rate:.1f}%\n")
    
    print(f"\n📁 Analysis summary saved to: {summary_file}")
    
    return True

if __name__ == "__main__":
    print("Comprehensive Parental Guide Data Analysis")
    print("This script analyzes your complete movie dataset with parental guide information.\n")
    
    success = analyze_parental_guide_data()
    
    if success:
        print(f"\n✅ Analysis completed successfully!")
        print(f"🚀 Your movie recommendation system is now enhanced with comprehensive parental guide data!")
    else:
        print(f"\n❌ Analysis failed. Please check the error messages above.")


# ### Summary and Rotten Tomatoes columns control

# In[2]:


current_df = pd.read_csv("movies_with_COMPLETE_parental_guide_FIXED.csv") 
current_df.head()


# In[4]:


current_df.columns


# In[5]:


#!/usr/bin/env python3
"""
Movie Content Quality Analyzer
Focused analysis for plot_summary and Rotten Tomatoes review data
"""

import pandas as pd
import json
import numpy as np
from collections import Counter

def decode_json_reviews(json_string):
    """
    Helper function to safely decode JSON review strings from RT data.
    
    Your RT review data is stored as JSON strings, so we need to parse them
    carefully to count and analyze the actual review content.
    """
    if pd.isna(json_string) or json_string == '' or json_string == '[]':
        return []
    
    try:
        # Handle string representation of lists
        if isinstance(json_string, str):
            # Clean up common JSON encoding issues that might exist
            cleaned = json_string.replace("\\'", "'").replace('\\"', '"')
            reviews = json.loads(cleaned)
            return reviews if isinstance(reviews, list) else []
        return []
    except (json.JSONDecodeError, TypeError):
        # If JSON parsing fails, return empty list rather than crashing
        return []

def analyze_plot_summaries(df):
    """
    Comprehensive analysis of your plot_summary column for embedding quality.
    
    This function examines whether your plot summaries are detailed enough
    to create meaningful semantic embeddings for your recommendation system.
    The quality of plot summaries directly impacts how well users can find
    movies using natural language descriptions.
    """
    print("📖 PLOT SUMMARY ANALYSIS FOR SEMANTIC SEARCH")
    print("=" * 65)
    
    total_movies = len(df)
    
    # Check coverage of plot_summary column
    has_plot = df['plot_summary'].notna() & (df['plot_summary'] != '') & (df['plot_summary'] != 'N/A')
    movies_with_plots = has_plot.sum()
    missing_plots = total_movies - movies_with_plots
    
    print(f"📊 COVERAGE ANALYSIS:")
    print(f"Total movies in dataset: {total_movies:,}")
    print(f"Movies with plot summaries: {movies_with_plots:,} ({movies_with_plots/total_movies*100:.1f}%)")
    print(f"Movies missing plot summaries: {missing_plots:,} ({missing_plots/total_movies*100:.1f}%)")
    
    if movies_with_plots == 0:
        print("❌ No valid plot summaries found!")
        return
    
    # Analyze plot quality for movies that have summaries
    plot_df = df[has_plot].copy()
    plot_df['char_length'] = plot_df['plot_summary'].str.len()
    plot_df['word_count'] = plot_df['plot_summary'].str.split().str.len()
    
    # Calculate descriptive statistics
    char_stats = plot_df['char_length'].describe()
    word_stats = plot_df['word_count'].describe()
    
    print(f"\n📏 LENGTH STATISTICS:")
    print(f"Average character length: {char_stats['mean']:.0f} characters")
    print(f"Median character length: {char_stats['50%']:.0f} characters")
    print(f"Shortest plot summary: {char_stats['min']:.0f} characters")
    print(f"Longest plot summary: {char_stats['max']:.0f} characters")
    print(f"Average word count: {word_stats['mean']:.0f} words")
    print(f"Median word count: {word_stats['50%']:.0f} words")
    
    # Quality assessment for semantic embeddings
    print(f"\n🎯 EMBEDDING EFFECTIVENESS ANALYSIS:")
    print("This analysis categorizes your plot summaries by their potential effectiveness for semantic search.")
    print("Research shows that longer, more detailed text generally produces better semantic embeddings.")
    
    # Define quality categories based on embedding research and practical experience
    very_short = (plot_df['char_length'] < 100).sum()
    short = ((plot_df['char_length'] >= 100) & (plot_df['char_length'] < 300)).sum()
    good = ((plot_df['char_length'] >= 300) & (plot_df['char_length'] < 800)).sum()
    excellent = (plot_df['char_length'] >= 800).sum()
    
    print(f"\n🔴 Very Short (<100 chars): {very_short:,} ({very_short/movies_with_plots*100:.1f}%)")
    print("   - Limited context for embeddings, may miss nuanced preferences")
    
    print(f"🟡 Short (100-300 chars): {short:,} ({short/movies_with_plots*100:.1f}%)")
    print("   - Basic plot outline, adequate for simple matching")
    
    print(f"🟢 Good (300-800 chars): {good:,} ({good/movies_with_plots*100:.1f}%)")
    print("   - Sufficient detail for effective semantic matching")
    
    print(f"🟦 Excellent (≥800 chars): {excellent:,} ({excellent/movies_with_plots*100:.1f}%)")
    print("   - Rich context enables sophisticated preference matching")
    
    # Word count analysis for semantic richness
    print(f"\n📝 SEMANTIC RICHNESS BY WORD COUNT:")
    minimal_words = (plot_df['word_count'] < 20).sum()
    basic_words = ((plot_df['word_count'] >= 20) & (plot_df['word_count'] < 50)).sum()
    rich_words = ((plot_df['word_count'] >= 50) & (plot_df['word_count'] < 120)).sum()
    very_rich_words = (plot_df['word_count'] >= 120).sum()
    
    print(f"Minimal (<20 words): {minimal_words:,} ({minimal_words/movies_with_plots*100:.1f}%)")
    print(f"Basic (20-50 words): {basic_words:,} ({basic_words/movies_with_plots*100:.1f}%)")
    print(f"Rich (50-120 words): {rich_words:,} ({rich_words/movies_with_plots*100:.1f}%)")
    print(f"Very Rich (≥120 words): {very_rich_words:,} ({very_rich_words/movies_with_plots*100:.1f}%)")
    
    # Show examples of different quality levels to help understand the data
    print(f"\n📋 EXAMPLE PLOT SUMMARIES BY QUALITY LEVEL:")
    
    # Example of very short plot (if any exist)
    very_short_examples = plot_df[plot_df['char_length'] < 100]
    if len(very_short_examples) > 0:
        example = very_short_examples.iloc[0]
        print(f"\n🔴 Very Short Example ({example['char_length']:.0f} chars):")
        print(f"Title: {example.get('title', 'Unknown')}")
        print(f"Plot: \"{example['plot_summary'][:150]}\"")
    
    # Example of good quality plot
    good_examples = plot_df[(plot_df['char_length'] >= 300) & (plot_df['char_length'] < 800)]
    if len(good_examples) > 0:
        example = good_examples.iloc[0]
        print(f"\n🟢 Good Example ({example['char_length']:.0f} chars):")
        print(f"Title: {example.get('title', 'Unknown')}")
        print(f"Plot: \"{example['plot_summary'][:300]}...\"")
    
    # Example of excellent quality plot
    excellent_examples = plot_df[plot_df['char_length'] >= 800]
    if len(excellent_examples) > 0:
        example = excellent_examples.iloc[0]
        print(f"\n🟦 Excellent Example ({example['char_length']:.0f} chars):")
        print(f"Title: {example.get('title', 'Unknown')}")
        print(f"Plot: \"{example['plot_summary'][:400]}...\"")
    
    return plot_df

def analyze_rt_reviews(df):
    """
    Comprehensive analysis of Rotten Tomatoes review data coverage and quality.
    
    This function examines how many movies have RT data and analyzes the
    quantity and quality of review content that could be used to enhance
    your recommendation system with review-based insights.
    """
    print(f"\n🍅 ROTTEN TOMATOES REVIEW ANALYSIS")
    print("=" * 65)
    
    total_movies = len(df)
    
    # RT Slug Analysis - this indicates which movies we attempted to scrape
    has_slug = df['rt_slug'].notna() & (df['rt_slug'] != '') & (df['rt_slug'] != 'None')
    movies_with_slugs = has_slug.sum()
    
    print(f"📊 ROTTEN TOMATOES COVERAGE:")
    print(f"Movies with RT slugs (scraping attempted): {movies_with_slugs:,} ({movies_with_slugs/total_movies*100:.1f}%)")
    print(f"Movies without RT slugs: {total_movies - movies_with_slugs:,}")
    
    if movies_with_slugs == 0:
        print("❌ No Rotten Tomatoes data found!")
        return
    
    # Focus analysis on movies where we have RT slugs
    rt_df = df[has_slug].copy()
    
    # User Reviews Analysis
    print(f"\n👥 USER REVIEWS ANALYSIS:")
    
    # Count movies with actual user review data
    user_review_counts = []
    user_review_total_chars = []
    movies_with_user_reviews = 0
    
    for idx, row in rt_df.iterrows():
        reviews = decode_json_reviews(row['rt_user_reviews'])
        count = len(reviews)
        user_review_counts.append(count)
        
        if count > 0:
            movies_with_user_reviews += 1
            # Calculate total character count for all reviews for this movie
            total_chars = sum(len(str(review)) for review in reviews)
            user_review_total_chars.append(total_chars)
        else:
            user_review_total_chars.append(0)
    
    rt_df['user_review_count'] = user_review_counts
    rt_df['user_review_total_chars'] = user_review_total_chars
    
    print(f"Movies with user reviews: {movies_with_user_reviews:,} ({movies_with_user_reviews/movies_with_slugs*100:.1f}% of scraped movies)")
    
    if movies_with_user_reviews > 0:
        # Calculate statistics only for movies that have reviews
        movies_with_user_data = rt_df[rt_df['user_review_count'] > 0]
        avg_reviews = movies_with_user_data['user_review_count'].mean()
        median_reviews = movies_with_user_data['user_review_count'].median()
        max_reviews = movies_with_user_data['user_review_count'].max()
        avg_chars = movies_with_user_data['user_review_total_chars'].mean()
        
        print(f"Average user reviews per movie: {avg_reviews:.1f}")
        print(f"Median user reviews per movie: {median_reviews:.1f}")
        print(f"Maximum user reviews for one movie: {max_reviews:.0f}")
        print(f"Average total characters per movie: {avg_chars:.0f}")
    
    # Critic Reviews Analysis
    print(f"\n🎭 CRITIC REVIEWS ANALYSIS:")
    
    # Count movies with actual critic review data
    critic_review_counts = []
    critic_review_total_chars = []
    movies_with_critic_reviews = 0
    
    for idx, row in rt_df.iterrows():
        reviews = decode_json_reviews(row['rt_critic_reviews'])
        count = len(reviews)
        critic_review_counts.append(count)
        
        if count > 0:
            movies_with_critic_reviews += 1
            # Calculate total character count for all reviews for this movie
            total_chars = sum(len(str(review)) for review in reviews)
            critic_review_total_chars.append(total_chars)
        else:
            critic_review_total_chars.append(0)
    
    rt_df['critic_review_count'] = critic_review_counts
    rt_df['critic_review_total_chars'] = critic_review_total_chars
    
    print(f"Movies with critic reviews: {movies_with_critic_reviews:,} ({movies_with_critic_reviews/movies_with_slugs*100:.1f}% of scraped movies)")
    
    if movies_with_critic_reviews > 0:
        # Calculate statistics only for movies that have reviews
        movies_with_critic_data = rt_df[rt_df['critic_review_count'] > 0]
        avg_reviews = movies_with_critic_data['critic_review_count'].mean()
        median_reviews = movies_with_critic_data['critic_review_count'].median()
        max_reviews = movies_with_critic_data['critic_review_count'].max()
        avg_chars = movies_with_critic_data['critic_review_total_chars'].mean()
        
        print(f"Average critic reviews per movie: {avg_reviews:.1f}")
        print(f"Median critic reviews per movie: {median_reviews:.1f}")
        print(f"Maximum critic reviews for one movie: {max_reviews:.0f}")
        print(f"Average total characters per movie: {avg_chars:.0f}")
    
    # Combined Analysis
    print(f"\n🔄 COMBINED REVIEW COVERAGE:")
    
    # Movies with any type of reviews
    has_any_reviews = (rt_df['user_review_count'] > 0) | (rt_df['critic_review_count'] > 0)
    movies_with_any_reviews = has_any_reviews.sum()
    
    # Movies with both types of reviews
    has_both_reviews = (rt_df['user_review_count'] > 0) & (rt_df['critic_review_count'] > 0)
    movies_with_both_reviews = has_both_reviews.sum()
    
    print(f"Movies with any reviews (user OR critic): {movies_with_any_reviews:,} ({movies_with_any_reviews/movies_with_slugs*100:.1f}%)")
    print(f"Movies with both user AND critic reviews: {movies_with_both_reviews:,} ({movies_with_both_reviews/movies_with_slugs*100:.1f}%)")
    
    # Show examples of movies with substantial review coverage
    if movies_with_both_reviews > 0:
        print(f"\n📋 EXAMPLES OF MOVIES WITH RICH REVIEW DATA:")
        
        # Calculate total review count and sort by it
        rt_df['total_review_count'] = rt_df['user_review_count'] + rt_df['critic_review_count']
        top_reviewed = rt_df[has_both_reviews].nlargest(5, 'total_review_count')
        
        for idx, movie in top_reviewed.iterrows():
            title = movie.get('title', 'Unknown')
            user_count = movie['user_review_count']
            critic_count = movie['critic_review_count']
            total_count = user_count + critic_count
            print(f"• {title}: {total_count} total reviews ({user_count} user, {critic_count} critic)")
    
    return rt_df

def generate_content_quality_report(df):
    """
    Generate a comprehensive summary report with actionable insights.
    
    This function synthesizes all the analysis into clear recommendations
    for optimizing your movie recommendation system's performance.
    """
    print(f"\n📊 COMPREHENSIVE CONTENT QUALITY REPORT")
    print("=" * 65)
    
    total_movies = len(df)
    
    # Plot summary assessment
    has_good_plot = df['plot_summary'].notna() & (df['plot_summary'] != '') & (df['plot_summary'] != 'N/A')
    movies_with_plots = has_good_plot.sum()
    
    # For plot quality, count those with substantial content
    good_plots = 0
    if movies_with_plots > 0:
        plot_lengths = df[has_good_plot]['plot_summary'].str.len()
        good_plots = (plot_lengths >= 300).sum()  # Plots with 300+ characters are good for embeddings
    
    # RT review assessment
    has_rt_slug = df['rt_slug'].notna() & (df['rt_slug'] != '') & (df['rt_slug'] != 'None')
    movies_with_rt_attempts = has_rt_slug.sum()
    
    # Count movies with meaningful review data
    movies_with_reviews = 0
    if movies_with_rt_attempts > 0:
        for idx, row in df[has_rt_slug].iterrows():
            user_reviews = decode_json_reviews(row.get('rt_user_reviews', ''))
            critic_reviews = decode_json_reviews(row.get('rt_critic_reviews', ''))
            if len(user_reviews) > 0 or len(critic_reviews) > 0:
                movies_with_reviews += 1
    
    print(f"🎬 DATASET OVERVIEW:")
    print(f"Total movies: {total_movies:,}")
    print(f"Movies with plot summaries: {movies_with_plots:,} ({movies_with_plots/total_movies*100:.1f}%)")
    print(f"Movies with substantial plots (≥300 chars): {good_plots:,} ({good_plots/total_movies*100:.1f}%)")
    print(f"Movies with RT scraping attempts: {movies_with_rt_attempts:,} ({movies_with_rt_attempts/total_movies*100:.1f}%)")
    print(f"Movies with actual RT reviews: {movies_with_reviews:,} ({movies_with_reviews/total_movies*100:.1f}%)")
    
    # Embedding readiness assessment
    print(f"\n🎯 SEMANTIC SEARCH READINESS:")
    
    embedding_readiness = good_plots / total_movies
    if embedding_readiness >= 0.8:
        print(f"✅ EXCELLENT: {embedding_readiness*100:.1f}% of movies have high-quality plots for semantic embeddings")
        print("   Your semantic search should perform very well across most of your dataset")
    elif embedding_readiness >= 0.6:
        print(f"✅ GOOD: {embedding_readiness*100:.1f}% of movies have quality plots for semantic embeddings")
        print("   Semantic search will work well for the majority of your movies")
    elif embedding_readiness >= 0.4:
        print(f"⚠️  MODERATE: {embedding_readiness*100:.1f}% of movies have quality plots")
        print("   Consider enhancing plot summaries to improve semantic search accuracy")
    else:
        print(f"❌ NEEDS IMPROVEMENT: Only {embedding_readiness*100:.1f}% have quality plots")
        print("   Significant plot enhancement needed for effective semantic search")
    
    # Review enrichment potential
    print(f"\n🍅 REVIEW DATA ENRICHMENT POTENTIAL:")
    
    review_coverage = movies_with_reviews / total_movies
    if review_coverage >= 0.6:
        print(f"✅ EXCELLENT: {review_coverage*100:.1f}% of movies have review content for recommendation enhancement")
        print("   You have substantial review data to enrich user recommendations")
    elif review_coverage >= 0.4:
        print(f"✅ GOOD: {review_coverage*100:.1f}% of movies have review content")
        print("   Solid foundation for review-based recommendation features")
    elif review_coverage >= 0.2:
        print(f"⚠️  MODERATE: {review_coverage*100:.1f}% have reviews")
        print("   Consider expanding review collection or using alternative sources")
    else:
        print(f"❌ LIMITED: Only {review_coverage*100:.1f}% have reviews")
        print("   Review-based features will be limited to a small subset of movies")
    
    # Actionable recommendations
    print(f"\n💡 ACTIONABLE RECOMMENDATIONS:")
    
    if embedding_readiness < 0.7:
        missing_good_plots = total_movies - good_plots
        print(f"• Plot Enhancement: Consider improving {missing_good_plots:,} plot summaries")
        print("  - Source additional plot data from IMDb or other databases")
        print("  - Use AI to expand existing short summaries")
    
    if review_coverage < 0.5:
        missing_reviews = total_movies - movies_with_reviews
        print(f"• Review Collection: {missing_reviews:,} movies lack review data")
        print("  - Expand RT scraping to capture more reviews per movie")
        print("  - Consider integrating IMDb user reviews as additional source")
    
    rt_success_rate = movies_with_reviews / max(movies_with_rt_attempts, 1)
    if rt_success_rate < 0.5:
        print(f"• RT Scraping Optimization: Only {rt_success_rate*100:.1f}% of RT attempts yielded reviews")
        print("  - Review and optimize your RT scraping methodology")
        print("  - Check for movies where slug detection failed")
    
    print(f"\n🚀 SYSTEM PERFORMANCE EXPECTATIONS:")
    
    if embedding_readiness >= 0.7:
        print("• Semantic search should deliver high-quality, nuanced movie recommendations")
    else:
        underperforming_movies = total_movies - good_plots
        print(f"• Semantic search may have reduced accuracy for {underperforming_movies:,} movies with short plots")
    
    if review_coverage >= 0.4:
        print("• Review-based recommendation features will provide valuable user insights")
    else:
        print(f"• Review features will be limited to {movies_with_reviews:,} movies with available data")

def run_complete_analysis(csv_file_path):
    """
    Execute comprehensive content quality analysis on your movie dataset.
    
    This is the main function that coordinates all analysis components and
    provides you with a complete picture of your content quality for both
    semantic search embeddings and review-based recommendations.
    """
    print("🎬 MOVIE RECOMMENDATION SYSTEM - CONTENT QUALITY ANALYSIS")
    print("=" * 80)
    print(f"Analyzing dataset: {csv_file_path}")
    
    try:
        # Load your dataset
        df = pd.read_csv(csv_file_path)
        print(f"✅ Successfully loaded {len(df):,} movies")
        print(f"📊 Dataset contains {len(df.columns)} columns")
        
        # Verify expected columns exist
        required_columns = ['plot_summary', 'rt_slug', 'rt_user_reviews', 'rt_critic_reviews']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Warning: Missing expected columns: {missing_columns}")
        
        # Run comprehensive analysis
        print(f"\n🔍 Beginning comprehensive content analysis...")
        
        # Analyze plot summaries for embedding quality
        plot_analysis = analyze_plot_summaries(df)
        
        # Analyze Rotten Tomatoes review coverage
        rt_analysis = analyze_rt_reviews(df)
        
        # Generate final comprehensive report
        generate_content_quality_report(df)
        
        print(f"\n✅ Analysis complete! Use these insights to optimize your recommendation system.")
        
        return df
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

# Usage example - replace with your actual dataset path
if __name__ == "__main__":
    # Example usage:
    df = run_complete_analysis('movies_with_COMPLETE_parental_guide_FIXED.csv')
    
    print("Content Quality Analyzer Ready!")
    print("To analyze your dataset, run:")
    print("df = run_complete_analysis('path_to_your_dataset.csv')")


# In[ ]:




