import requests

ENDPOINT = "https://mldevops-p4-model-deploy.onrender.com/predict"

# Define the JSON data to be sent in the request
json_data = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States"
}

# Make the POST request
response = requests.post(ENDPOINT, json=json_data)

# Print the response
print("Response Code:", response.status_code)
print("Response JSON:", response.json())
