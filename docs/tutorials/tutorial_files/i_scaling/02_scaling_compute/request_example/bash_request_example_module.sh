curl -X POST \
        "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '[{"text_output": "### Instruction: Explain three ways to reduce carbon
         emissions. ### Response:"}]
           '