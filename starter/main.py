import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from starter.ml.data import process_data

from starter.utils import cat_features, path_model

# Define the FastAPI app
app = FastAPI()

# Load the machine learning model
fln_model = path_model / "model.pickle"
fln_ohe_encoder = path_model / "encoder.pickle"
fln_label_encoder = path_model / "label_encoder.pickle"

with open(fln_model, "rb") as file:
    model = pickle.load(file)
with open(fln_ohe_encoder, "rb") as file:
    encoder = pickle.load(file)
with open(fln_label_encoder, "rb") as file:
    label_encoder = pickle.load(file)


# Pydantic model for request body
class Example(BaseModel):
    age: float = 52
    workclass: str = "Self-emp-not-inc"
    fnlgt: float = 209642
    education: str = "HS-grad"
    education_num: float = Field(alias="education-num", default=9)
    marital_status: str = Field(alias="marital-status",
                                default="Married-civ-spouse")
    occupation: str = "Exec-managerial"
    relationship: str = "Husband"
    race: str = "White"
    sex: str = "Male"
    capital_gain: float = Field(alias="capital-gain", default=0)
    capital_loss: float = Field(alias="capital-loss", default=0)
    hours_per_week: float = Field(alias="hours-per-week", default=45)
    native_country: str = Field(alias="native-country",
                                default="United-States")

    class Config:
        schema_extra = {
                "examples": [
                    {
                        "age": 39,
                        "workclass": "State-gov",
                        "fnlgt": 77516,
                        "education": "Bachelors",
                        "education-num": 13,
                        "marital-status": "Never-married",
                        "occupation": "Adm-clerical",
                        "relationship": "Not-in-family",
                        "race": "White",
                        "sex": "Male",
                        "capital-gain": 2174,
                        "capital-loss": 0,
                        "hours-per-week": 40,
                        "native-country": "United-States"
                    }
                ]
            }


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML model census API"}


@app.post("/predict", response_model_by_alias=True)
def predict(data: Example):
    try:
        # Prepare data for prediction
        input_values = pd.DataFrame([data.dict(by_alias=True)])
        Xs, _, _, _ = process_data(
            input_values,
            categorical_features=cat_features,
            label=None, training=False, encoder=encoder, lb=None
        )

        prediction = label_encoder.inverse_transform(model.predict(Xs))
        print(prediction)

        return {"prediction": prediction[0]}  # Assuming a single prediction
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
