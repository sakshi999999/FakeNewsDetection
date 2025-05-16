import requests

def fetch_latest_headlines(country="us", category=None):
    """
    Fetch the latest headlines from the News API.
    
    Args:
        country (str): The 2-letter ISO 3166-1 code of the country (default: "us")
        category (str, optional): Category of news (e.g., "technology", "business")
        
    Returns:
        list: A list of headlines
    """
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": country,
        "apiKey": "089ce24de3e84465b1f78eb15fff1f86"  # Replace with your NewsAPI key
    }
    
    if category:
        params["category"] = category
        
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("status") == "ok":
            articles = data["articles"]
            return [{"title": article["title"], 
                     "source": article.get("source", {}).get("name", "Unknown"),
                     "url": article.get("url", ""),
                     "description": article.get("description", "")} 
                    for article in articles]
        else:
            print(f"Error from API: {data.get('message', 'Unknown error')}")
            return []
    except Exception as e:
        print("Error fetching headlines:", e)
        return []

def get_article_content(url):
    """
    A simple function to get some content from a news URL.
    Note: This is a very basic implementation and won't work for most news sites
    that use JavaScript or have paywalls. A proper implementation would require
    more sophisticated web scraping techniques.
    """
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            # Return just the first 500 characters as a simple example
            text = response.text
            start_idx = text.find("<body")
            if start_idx > 0:
                text = text[start_idx:]
            # Remove HTML tags very simplistically
            text = text.replace("<", " <").replace(">", "> ")
            import re
            text = re.sub(r'<[^>]*>', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:2000] + "..." if len(text) > 2000 else text
        return "Could not retrieve article content."
    except Exception as e:
        return f"Error retrieving content: {str(e)}"
