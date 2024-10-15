import os
import re
from flask import Flask, render_template, request
import requests
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Prevent Transformers from using TensorFlow or Flax (must be set before importing transformers)
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'

# Import after setting environment variables
from transformers import pipeline

# Create Flask app
app = Flask(__name__)

# Initialize the summarizer pipeline with PyTorch
# Specify the model to ensure consistency
summarizer = pipeline('summarization', model='facebook/bart-large-cnn', framework='pt')

# Fetch the NYT API key from environment variables
NYT_API_KEY = os.getenv('NYT_API_KEY')

def fetch_articles(query, page=0):
    if not NYT_API_KEY:
        return {"error": "NYT API key is missing. Please set NYT_API_KEY in your environment variables."}
    try:
        url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
        params = {
            'q': query,
            'api-key': NYT_API_KEY,
            'page': page,
        }
        response = requests.get(url, params=params)
        if response.status_code == 429:
            return {"error": "Rate limit exceeded. Please try again later."}
        response.raise_for_status()
        data = response.json()
        # Check for errors in the response
        if 'fault' in data:
            return {"error": data['fault']['faultstring']}
        articles = data.get('response', {}).get('docs', [])
        return {"articles": articles}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def get_article_content(article):
    content_parts = [
        article.get('abstract', ''),
        article.get('snippet', ''),
        article.get('lead_paragraph', '')
    ]
    # Filter out empty strings and combine the content
    content = ' '.join([part for part in content_parts if part])
    return content

def clean_text(text):
    # Remove HTML tags
    clean = re.sub(r'<.*?>', '', text)
    # Remove special characters
    clean = re.sub(r'[^A-Za-z0-9\s.,!?\'"-]', '', clean)
    return clean

def summarize_text(text, summarizer):
    text = clean_text(text)
    words = text.split()
    text_length = len(words)
    
    # Remove debugging print statements if not needed
    # print(f"Text length: {text_length} words")
    # print(f"Text content:\n{text}\n")
    
    if text_length < 50:
        return text  # Return the original text if too short
    
    # Determine max_length based on input_length
    # Ensure that max_length is less than input_length
    # For example, set max_length to 50% of input_length or use a fixed ratio
    # Here, we set max_length to min(130, text_length - 10)
    max_length = min(130, text_length - 10)
    min_length = 30
    
    # Ensure that min_length is less than max_length
    if min_length >= max_length:
        min_length = max_length // 2  # Adjust to half of max_length
    
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        # Log the exception if needed
        print(f"Error during summarization: {e}")
        return text  # Return the original text if summarization fails

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    query = request.form.get('query')
    page = int(request.form.get('page', 0))
    result = fetch_articles(query, page)

    if "error" in result:
        return render_template('index.html', error=result["error"])

    articles = result["articles"]
    summarized_articles = []

    for article in articles:
        headline = article.get('headline', {}).get('main', 'No Title')
        web_url = article.get('web_url', '')
        content = get_article_content(article)
        if content:
            summary = summarize_text(content, summarizer)
        else:
            summary = "No content available to summarize."

        summarized_articles.append({
            'title': headline,
            'url': web_url,
            'summary': summary
        })

    return render_template('results.html', articles=summarized_articles, query=query, page=page)

if __name__ == '__main__':
    app.run(debug=True)