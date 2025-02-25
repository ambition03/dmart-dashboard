import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from fastapi import FastAPI
import uvicorn

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Convert date columns to datetime
    date_columns = ["First Trx Date", "Last Trx Date", "created_at"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Drop unnecessary columns
    drop_cols = ["Chain", "Store Address", "Software Version", "First Trx Date", "Last Trx Date", "status"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
    
    # Handle missing values
    df.fillna("Unknown", inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ["Store", "State", "City", "Pin Code", "Terminal Category", "issue_type", "cf_issue_sub_category"]
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Convert all columns to numeric explicitly
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values with median for numerical columns
    df.fillna(df.median(), inplace=True)
    
    # Define target variable (predicting issue frequency per store)
    df["Target"] = df.groupby("Store")["ticket_id"].transform("count")
    
    return df, label_encoders

# Train the model
def train_model(df):
    X = df.drop(columns=["Target"])
    y = df["Target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=10, 
        min_samples_leaf=5, 
        max_features='sqrt', 
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    joblib.dump(model, "dmart_issue_predictor.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    print("Model training complete.")
    return model, scaler

# Load trained model
try:
    model = joblib.load("dmart_issue_predictor.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    model, scaler = None, None

# FastAPI chatbot application
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the D-Mart Operations Chatbot! Ask anything about D-Mart issues."}

@app.get("/query")
def query_data(query: str):
    if "highest issues" in query.lower():
        top_store = df.groupby("Store")["Target"].sum().idxmax()
        return {"response": f"The store with the highest number of issues is {top_store}."}
    elif "issues in last month" in query.lower():
        # last_month = df[df["created_at"] >= pd.to_datetime("now") - pd.DateOffset(months=1)]
        df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')  # Convert to datetime
        df = df.dropna(subset=["created_at"])  # Remove rows with invalid dates

        last_month = df[df["created_at"] >= pd.to_datetime("today") - pd.DateOffset(months=1)]

        return {"response": f"There were {len(last_month)} issues reported in the last month."}
    elif "store" in query.lower():
        store_name = query.split("store")[-1].strip()
        store_issues = df[df["Store"] == store_name]
        return {"response": f"Store {store_name} has {len(store_issues)} reported issues."}
    else:
        return {"response": "I couldn't understand your question. Please ask about D-Mart operations!"}

@app.get("/predict")
def predict_issues(store_id: int):
    if model is None or scaler is None:
        return {"error": "Model not trained yet!"}
    
    store_data = df[df["Store"] == store_id].drop(columns=["Target"]).iloc[:1]
    if store_data.empty:
        return {"error": "Store ID not found in dataset!"}
    
    store_data = store_data.apply(pd.to_numeric, errors='coerce')
    store_data.fillna(store_data.median(), inplace=True)
    predicted_issues = model.predict(scaler.transform(store_data))[0]
    
    return {"store_id": store_id, "predicted_issues": int(predicted_issues)}

# if __name__ == "__main__":
#     file_path = "fd_tickets_202502181901.csv"
#     df, label_encoders = load_and_preprocess_data(file_path)
#     if model is None:
#         model, scaler = train_model(df)
    
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import asyncio

if __name__ == "__main__":
    file_path = "fd_tickets_202502181901.csv"
    df, label_encoders = load_and_preprocess_data(file_path)
    if model is None:
        model, scaler = train_model(df)
    
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)

    if asyncio.get_event_loop().is_running():
        print("Running inside an active event loop. Use `!uvicorn your_script:app --reload` in Jupyter Notebook or VS Code terminal.")
    else:
        server.run()
