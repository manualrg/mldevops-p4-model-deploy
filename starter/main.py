# Put the code for your API here.
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from starter.ml.data import process_data

from starter.utils import cat_features, path_model

# Define the FastAPI app
app = FastAPI()

# Load the machine learning model
#path_model = Path() / "model"
fln_model = path_model / "model.pickle"
fln_ohe_encoder =  path_model /  "encoder.pickle"
fln_label_encoder = path_model /  "label_encoder.pickle"



with open(fln_model, "rb") as file:
    model = pickle.load(file)
with open(fln_ohe_encoder, "rb") as file:
    encoder = pickle.load(file)
with open(fln_label_encoder, "rb") as file:
    label_encoder = pickle.load(file)



# Pydantic model for request body
class Example(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(alias="capital-gain")
    capital_loss: float = Field(alias="capital-loss")
    hours_per_week: float = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML model census API"}

@app.post("/predict", response_model_by_alias=True)
def predict(data: Example):
    try:
        # Prepare data for prediction
        input_values = pd.DataFrame([data.dict(by_alias=True)])  # Adjust based on your model's input
        Xs, _, _, _ = process_data(input_values, 
            categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=None
        )


        prediction = label_encoder.inverse_transform(model.predict(Xs))
        print(prediction)

        return {"prediction": prediction[0]}  # Assuming a single prediction
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
