from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app instance from main.py
import pytest

@pytest.fixture
def get_le50k_example():
    return {"age": 39,
         "workclass": "State-gov",
         "fnlgt": 77516,
         "education": "Bachelors",
         "education-num": 13,
         "marital-status": "Never-married",
         "occupation":"Adm-clerical", 
         "relationship": "Not-in-family",
         "race": "White",
         "sex": "Male",
         "capital-gain": 2174,
         "capital-loss": 0,
         "hours-per-week": 40,
         "native-country": "United-States",
         "salary": "<=50K"}
    

@pytest.fixture
def get_g50k_example():
    return {"age": 52,
         "workclass": "Self-emp-not-inc",
         "fnlgt": 209642,
         "education": "HS-grad",
         "education-num": 9,
         "marital-status": "Married-civ-spouse",
         "occupation":"Exec-managerial", 
         "relationship": "Husband",
         "race": "White",
         "sex": "Male",
         "capital-gain": 0,
         "capital-loss": 0,
         "hours-per-week": 45,
         "native-country": "United-States",
        "salary": ">50K"}



client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_post_prediction_le50k(get_le50k_example):
    test_data = get_le50k_example

    _ = test_data.pop("salary")
    print("@@@@@@@@@@@@@", test_data)
    
    response = client.post("/predict", json=test_data)

    print("@@@@@@@@@", response.json()["prediction"])
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert  response.json()["prediction"] in ["<=50K", ">50K"]


def test_post_prediction_g50k(get_g50k_example):
    test_data = get_g50k_example  
    expected = test_data.pop("salary")
    
    response = client.post("/predict", json=test_data)
    print("@@@@@@@@@", response.json()["prediction"])
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert  response.json()["prediction"] in ["<=50K", ">50K"]
