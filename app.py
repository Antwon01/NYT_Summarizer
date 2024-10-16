import os
import re
import logging  # Import the logging module
from flask import Flask, render_template, request
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

# Prevent Transformers from using TensorFlow or Flax (must be set before importing transformers)
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'

# Import after setting environment variables
from transformers import pipeline

# Create Flask app
app = Flask(__name__)

# Initialize the summarizer as None for lazy loading
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        logging.debug("Initializing the summarization pipeline...")
        summarizer = pipeline('summarization', model='facebook/bart-large-cnn', framework='pt')
        logging.debug("Summarization pipeline initialized.")
    return summarizer

# Fetch the NYT API key from environment variables
NYT_API_KEY = os.getenv('NYT_API_KEY')

def fetch_articles(query, page=0):
    logging.debug(f"Fetching articles for query: '{query}', page: {page}")
    if not NYT_API_KEY:
        error_msg = "NYT API key is missing. Please set NYT_API_KEY in your environment variables."
        logging.error(error_msg)
        return {"error": error_msg}
    try:
        url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
        params = {
            'q': query,
            'api-key': NYT_API_KEY,
            'page': page,
        }
        response = requests.get(url, params=params)
        if response.status_code == 429:
            error_msg = "Rate limit exceeded. Please try again later."
            logging.warning(error_msg)
            return {"error": error_msg}
        response.raise_for_status()
        data = response.json()
        # Check for errors in the response
        if 'fault' in data:
            error_msg = data['fault']['faultstring']
            logging.error(f"API fault: {error_msg}")
            return {"error": error_msg}
        articles = data.get('response', {}).get('docs', [])
        logging.debug(f"Retrieved {len(articles)} articles.")
        return {"articles": articles}
    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException while fetching articles: {e}")
        return {"error": str(e)}

def get_article_content(article):
    logging.debug("Extracting content from article.")
    content_parts = [
        article.get('abstract', ''),
        article.get('snippet', ''),
        article.get('lead_paragraph', '')
    ]
    # Filter out empty strings and combine the content
    content = ' '.join([part for part in content_parts if part])
    logging.debug(f"Extracted content length: {len(content)} characters.")
    return content

def clean_text(text):
    logging.debug("Cleaning text.")
    # Remove HTML tags
    clean = re.sub(r'<.*?>', '', text)
    # Remove special characters
    clean = re.sub(r'[^A-Za-z0-9\s.,!?\'"-]', '', clean)
    return clean

def summarize_text(text):
    logging.debug("Starting text summarization.")
    text = clean_text(text)
    words = text.split()
    text_length = len(words)
    logging.debug(f"Text length after cleaning: {text_length} words.")
    
    if text_length < 50:
        logging.debug("Text too short to summarize. Returning original text.")
        return text  # Return the original text if too short
    
    # Determine max_length based on input_length
    max_length = min(130, text_length - 10)
    min_length = 30
    
    # Ensure that min_length is less than max_length
    if min_length >= max_length:
        min_length = max_length // 2  # Adjust to half of max_length
        logging.debug(f"Adjusted min_length to {min_length}.")

    try:
        summarizer_instance = get_summarizer()
        summary = summarizer_instance(text, max_length=max_length, min_length=min_length, do_sample=False)
        logging.debug("Summarization successful.")
        return summary[0]['summary_text']
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return text  # Return the original text if summarization fails

@app.route('/', methods=['GET'])
def index():
    logging.debug("Index page accessed.")
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    query = request.form.get('query')
    page = int(request.form.get('page', 0))
    logging.debug(f"Summarize route called with query: '{query}', page: {page}")
    result = fetch_articles(query, page)

    if "error" in result:
        logging.error(f"Error fetching articles: {result['error']}")
        return render_template('index.html', error=result["error"])

    articles = result["articles"]
    summarized_articles = []

    for article in articles:
        headline = article.get('headline', {}).get('main', 'No Title')
        web_url = article.get('web_url', '')
        content = get_article_content(article)
        if content:
            summary = summarize_text(content)
        else:
            logging.warning(f"No content available for article titled '{headline}'.")
            summary = "No content available to summarize."

        summarized_articles.append({
            'title': headline,
            'url': web_url,
            'summary': summary
        })

    logging.debug("Rendering results page.")
    return render_template('results.html', articles=summarized_articles, query=query, page=page)

if __name__ == '__main__':
    app.run(debug=True)