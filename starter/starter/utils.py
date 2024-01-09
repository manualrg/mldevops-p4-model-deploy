from pathlib import Path

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
target = "salary"


path_root = (Path(__file__) 
             .parent  
             .parent  # starter
    )             
path_data = path_root / "data"
path_model = path_root / "model"
