curl -X POST \
        "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '[{"proteins_tabular": {"classification": "HYDROLASE"},
         "protein_sequence": ""}]'