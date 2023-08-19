# ----- Load key libraries and packages
import gradio as gr
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb


# ----- Useful lists
expected_inputs = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
categoricals = ["Sex", "Embarked"]
columns_to_scale = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

# ----- Helper Functions
# Function to load ML toolkit
def load_ml_toolkit(file_path=r"LP6_toolkit"):
    """
    This function loads the ML items into the app. It takes the path to the ML items to load it.

    Args:
        file_path (regexp, optional): It receives the file path to the ML items. The full default relative path is r"LP6_toolkit".

    Returns:
        file: It returns the pickle file (which in this case contains the Machine Learning items.)
    """
    
    with open(file_path, "rb") as file:
        loaded_toolkit = pickle.load(file)
    return loaded_toolkit


# Importing the toolkit
loaded_toolkit = load_ml_toolkit(r"LP6_toolkit")

# Instantiate the elements in the toolkit
encoder = loaded_toolkit["encoder"]
model = loaded_toolkit["model"]
scaler = loaded_toolkit["scaler"]


# Function to process inputs and return prediction
def process_and_predict(*args, encoder=encoder, scaler=scaler, model=model):
    """
    This function processes the inputs and returns the predicted survival status of the customer.
    It receives the user inputs, the encoder, scaler and model. The inputs are then put through the same process as was done during modelling, i.e. encode categoricals, scale columns, and return output.

    Args:
        encoder (OneHotEncoder, optional): It is the encoder used to encode the categorical features before training the model, and should be loaded either as part of the ML items or as a standalone item. Defaults to encoder, which comes with the ML Items dictionary.
        scaler (MinMaxScaler, optional): It is the scaler (MinMaxScaler) used to scale the numeric features before training the model, and should be loaded either as part of the ML Items or as a standalone item. Defaults to scaler, which comes with the ML Items dictionary.
        model (LightGBM, optional): This is the model that was trained and is to be used for the prediction. Since LightGBM seems to have issues with Pickle, import as a standalone. It defaults to "model", as loaded.

    Returns:
        Prediction (label): Returns the label of the predicted class, i.e. one of whether the given passenger is likely to survive or not.
    """
    
    # Convert inputs into a DataFrame
    input_data = pd.DataFrame([args], columns=expected_inputs)

    # Encode the categorical columns
    encoded_categoricals = encoder.transform(input_data[categoricals])
    encoded_categoricals = pd.DataFrame(encoded_categoricals, columns=encoder.get_feature_names_out().tolist())
    df_processed = input_data.join(encoded_categoricals)
    df_processed.drop(columns=categoricals, inplace=True)

    # Scale the numeric columns
    df_processed[columns_to_scale] = scaler.transform(df_processed[columns_to_scale])

    # Make the prediction
    model_output = model.predict(df_processed)
    return {"Prediction: Survived": float(model_output[0]), "Prediction: Did not Survive": 1-float(model_output[0])}


# ----- App Interface
# Inputs
# Whether the customer is a male or a female
Pclass = gr.Slider(label="Passenger Class", minimum=1, step=1, maximum=3, interactive=True, value=1) # Number of months the customer has stayed with the company
Sex = gr.Dropdown(label="Sex", choices=["female", "male"], value="female")
Age = gr.Slider(label="Age", minimum=1, step=1, interactive=True, value=1)
SibSp = gr.Slider(label="Number of Siblings/Spouses aboard", minimum=0, step=1, interactive=True, value=0, maximum=8)
Parch = gr.Slider(label="Number of Parents/Children aboard", minimum=0, step=1, interactive=True, value=0, maximum=6)
Fare = gr.Slider(label="Fare", minimum=0, step=1, interactive=True, value=0, maximum=1000)
Embarked = gr.Radio(label="Where the passenger embarked from", choices=["C", "Q", "S"], value="C")

# Output
gr.Interface(inputs=[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked],
            outputs = gr.Label("Awaiting Submission..."),
            fn=process_and_predict,
            title= "Titanic Passenger Survival Prediction App", 
            description= """This app uses a machine learning model to predict whether or not a passenger will survive the Titanic accident based on inputs made by you, the user. The (LightGBM) model was trained and built based on the Titanic Dataset."""
            ).launch(inbrowser= True,
                     show_error= True)