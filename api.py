# Import key libraries and packages
from fastapi import FastAPI
import pickle
import uvicorn
from pydantic import BaseModel
import pandas as pd


# Define key variables
toolkit_path = "LP6_toolkit"
expected_inputs = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
categoricals = ["Sex", "Embarked"]
columns_to_scale = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

# Function to load ML toolkit
def load_ml_toolkit(file_path= toolkit_path):
    """
    This function loads the ML items into this app by taking the path to the ML items.

    Args:
        file_path (regexp, optional): It receives the file path to the ML items, but defaults to the "src" folder in the repository.

    Returns:
        file: It returns the pickle file (which in this case contains the Machine Learning items.)
    """

    with open(file_path, "rb") as file:
        loaded_toolkit = pickle.load(file)
    return loaded_toolkit


# API base configuration
app = FastAPI(title="Titanic Survivors Prediction API",
              version="0.1",
              description="Titanic Survivors Prediction API"
              )

# Import the toolkit
loaded_toolkit = load_ml_toolkit()
encoder = loaded_toolkit["encoder"]
scaler = loaded_toolkit["scaler"]
model = loaded_toolkit["model"]

# Setup the BaseModel
class ModelInput(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_male: float
    Embarked_Q: float
    Embarked_S: float
    
def processing_FE(dataset,  imputer=None, FE=None):
    if imputer is not None:
        output_dataset = imputer.transform(dataset)
    else:
        output_dataset = dataset.copy()
    if FE is not None:
        output_dataset = FE.transform(output_dataset)
    return output_dataset


def make_prediction(Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S):
    df = pd.DataFrame([[Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S]],
                      columns = [Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S]
                      )
    # Scale the numeric columns
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    # Make the prediction and return output
    model_output = model.predict(df).tolist()
    return model_output

# Endpoints
@app.post("/survival")
async def predict(input: ModelInput):
    output_pred = make_prediction(
        Pclass=input.Pclass,
        Age=input.Age,
        SibSp=input.SibSp,
        Parch=input.Parch,
        Fare=input.Fare,
        Sex_male=input.Sex_male,
        Embarked_Q=input.Embarked_Q,
        Embarked_S=input.Embarked_S
        )

    # Labelling Model output
    if output_pred == 0:
        output_pred = "This person is UNLIKELY to survive"
    else:
        output_pred = "Positive. This person is LIKELY to survive"
    #return output_pred
    return {"prediction": output_pred,
            "input": input
            }

# Set the API to run
if __name__ == "__main__":
    uvicorn.run("api:app",
                reload=True
                )
