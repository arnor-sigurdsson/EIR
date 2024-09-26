curl -X POST \
"http://localhost:8000/predict" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '[{
    "power_input": "19 3 10 11 14 3 3 4 4 9 39 27 12 5 20 20 38 39 41 61 52 31 43 31",
    "power_output": ""
}]'
