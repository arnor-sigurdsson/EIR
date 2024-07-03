curl -X POST \
        "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '[{"imdb_reviews_windowed": "This movie was great! I loved it!",
        "imdb_reviews_longformer": "This movie was great! I loved it!",
        "imdb_reviews_tiny_bert": "This movie was great! I loved it!"}]'
        