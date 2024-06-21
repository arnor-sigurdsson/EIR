curl -X POST \
        "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '[{"imdb_reviews": "This movie was great! I loved it!"}]
           '