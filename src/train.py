import pickle
import os
import pandas as pd

def train_model(model,x_train,y_trian):
    model.fit(x_train,y_trian)
    return model

def save_model(model, name, path):
    os.makedirs(path, exist_ok=True)  # ensure folder exists

    file_path = os.path.join(path, f"{name}.pkl")

    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model Saved to {file_path}")

def load_model(path):
    with open(path,'rb')as f:
        return pickle.load(f)