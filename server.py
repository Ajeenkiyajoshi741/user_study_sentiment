import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews_df = pd.read_csv('data/reviews.csv')
reviews = reviews_df.to_dict('records')

# Add sentiment scores to the DataFrame
for index, row in reviews_df.iterrows():
    sentiment_scores = sia.polarity_scores(row['ReviewBody'])
    for score_key, score_value in sentiment_scores.items():
        reviews_df.at[index, score_key] = score_value

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        return sia.polarity_scores(review_body)

    def filter_reviews(self, location=None, start_date=None, end_date=None):
        filtered_df = reviews_df.copy()

        if location:
            filtered_df = filtered_df[filtered_df['Location'] == location]

        if start_date:
            filtered_df = filtered_df[filtered_df['Timestamp'] >= start_date]

        if end_date:
            filtered_df = filtered_df[filtered_df['Timestamp'] <= end_date]

        return filtered_df.sort_values(by='compound', ascending=False)

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ['QUERY_STRING'])
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            valid_locations = [
                'Albuquerque, New Mexico', 'Carlsbad, California', 'Chula Vista, California',
                'Colorado Springs, Colorado', 'Denver, Colorado', 'El Cajon, California',
                'El Paso, Texas', 'Escondido, California', 'Fresno, California',
                'La Mesa, California', 'Las Vegas, Nevada', 'Los Angeles, California',
                'Oceanside, California', 'Phoenix, Arizona', 'Sacramento, California',
                'Salt Lake City, Utah', 'San Diego, California', 'Tucson, Arizona'
            ]

            if location and location not in valid_locations:
                response_body = json.dumps({"error": "Invalid location"}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            filtered_df = self.filter_reviews(location, start_date, end_date)
            reviews_list = filtered_df.to_dict(orient='records')

            response_body = json.dumps(reviews_list, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                post_data = environ['wsgi.input'].read(content_length)
                post_params = parse_qs(post_data.decode('utf-8'))

                review_body = post_params.get('ReviewBody', [None])[0]
                location = post_params.get('Location', [None])[0]

                if not review_body or not location:
                    response_body = json.dumps({"error": "ReviewBody and Location are required"}).encode("utf-8")
                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]

                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                sentiment_scores = self.analyze_sentiment(review_body)

                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp,
                    **sentiment_scores
                }

                reviews_df.loc[len(reviews_df)] = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp,
                    **sentiment_scores
                }

                reviews_df.to_csv('data/reviews_with_sentiment.csv', index=False)
                response_body = json.dumps(new_review).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            except Exception as e:
                response_body = json.dumps({"error": str(e)}).encode("utf-8")
                start_response("500 Internal Server Error", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
