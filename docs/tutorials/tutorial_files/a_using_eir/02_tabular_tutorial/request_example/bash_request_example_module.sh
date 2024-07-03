curl -X POST \
        "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '[{"poker_hands": {"S1": "3", "C1": "12", "S2": "3", "C2": "2", "S3": "3",
         "C3": "11", "S4": "4", "C4": "5", "S5": "2", "C5": "5"}}]'