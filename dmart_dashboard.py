import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
import spacy
import os

import os
import spacy

# Check if spaCy model is installed, otherwise download it
spacy_model = "en_core_web_sm"

try:
    nlp = spacy.load(spacy_model)
except OSError:
    os.system(f"python -m spacy download {spacy_model}")
    nlp = spacy.load(spacy_model)


# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("dmart_issue_predictor.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("fd_tickets_202502181901.csv")
    df.columns = df.columns.str.strip()
    if "created_at" not in df.columns:
        raise KeyError("Column 'created_at' is missing from the dataset. Available columns: " + str(df.columns.tolist()))
    df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')
    df.dropna(subset=["created_at"], inplace=True)
    return df

model, scaler = load_model()
df = load_data()

st.title("🛒 D-Mart Operations Chatbot")
st.write("Ask anything about D-Mart issues, trends, categories, predictions, or analysis!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def predict_issues(store_id):
    store_data = df[df["Store"] == store_id].drop(columns=["ticket_id"], errors="ignore").iloc[:1]
    if store_data.empty:
        return "Store ID not found in dataset!"
    store_data = store_data.apply(pd.to_numeric, errors='coerce')
    store_data.fillna(store_data.median(), inplace=True)
    predicted_issues = model.predict(scaler.transform(store_data))[0]
    return f"Predicted number of issues for Store {store_id}: {int(predicted_issues)}"

def calculate_aso_requirements():
    df["issues_per_aso"] = df["ticket_id"].groupby(df["City"]).transform("count") // 10  # Assuming each ASO handles 10 issues
    city_aso_requirements = df.groupby("City")["issues_per_aso"].max().to_dict()
    return city_aso_requirements

def checklist_for_aso():
    return [
        "Check all reported issues from the store log.",
        "Inspect hardware and software systems.",
        "Ensure proper connectivity and power backup.",
        "Conduct training for store staff on issue resolution.",
        "Verify previous unresolved issues and take necessary actions.",
        "Document findings and submit a report."
    ]



def generate_plot(query):
    df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')
    
    if "trend" in query or "monthly" in query:
        df["month_year"] = df["created_at"].dt.to_period("M")
        monthly_issues = df.groupby("month_year")["ticket_id"].count()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=monthly_issues.index.astype(str), y=monthly_issues.values, ax=ax)
        ax.set_title("Monthly Issue Trend")
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Number of Issues")
        st.pyplot(fig)



def process_query(query):
    query = query.lower()
    doc = nlp(query)
    keywords = [token.text for token in doc if token.is_alpha]
    response = ""
    
    if "aso" in query and "city" in query:
        city_aso_requirements = calculate_aso_requirements()
        response += "Estimated ASOs required per city:\n" + "\n".join([f"{k}: {v} ASOs" for k, v in city_aso_requirements.items()])
    
    if "checklist" in query and "aso" in query:
        checklist = checklist_for_aso()
        response += "Checklist for ASO at store:\n" + "\n".join([f"- {task}" for task in checklist])
    
    if "location" in query or "city" in query or "state" in query:
        if "highest" in query or "most" in query:
            top_location = df.groupby("City")["ticket_id"].count().idxmax()
            response += f"City with the highest number of issues: {top_location}\n"
        elif "total" in query:
            location_counts = df.groupby("City")["ticket_id"].count().to_dict()
            response += "Total issues per city:\n" + "\n".join([f"{k}: {v}" for k, v in location_counts.items()])
        else:
            response += "Please specify if you need highest issues, total issues, or trends for a location."
    
    # if "category" in query:
    #     categories = df["issue_type"].dropna().unique()
    #     response += f"Total unique issue categories: {len(categories)}\nCategories: {', '.join(map(str, categories))}\n"

    if "category" in query:
        category_counts = df.groupby("issue_type")["ticket_id"].count().to_dict()
        response += "Category-wise ticket count:\n" + "\n".join([f"{k}: {v}" for k, v in category_counts.items()])
    
    
    if "store" in query:
        store_counts = df["Store"].value_counts()
        response += f"Top store with most issues: {store_counts.idxmax()} ({store_counts.max()} issues)\n"
    
    # if "open" in query:
    #     open_tickets = df[df["status"].str.lower() == "open"]
    #     response += f"Open tickets: {len(open_tickets)}\n"


    if "open" in query and "category" in query:
        open_category_counts = df[df["status"].str.lower() == "open"].groupby("issue_type")["ticket_id"].count().to_dict()
        response += "Open tickets by category:\n" + "\n".join([f"{k}: {v}" for k, v in open_category_counts.items()])
    
    if "checklist" in query and "aso" in query:
     checklist = [
        "Check all reported issues from the store log.",
        "Inspect hardware and software systems.",
        "Ensure proper connectivity and power backup.",
        "Conduct training for store staff on issue resolution.",
        "Verify previous unresolved issues and take necessary actions.",
        "Document findings and submit a report."
        ]
     response += "Checklist for ASO at store:\n" + "\n".join([f"- {task}" for task in checklist])


    if "aso" in query and "city" in query:
     df["issues_per_aso"] = df["ticket_id"].groupby(df["City"]).transform("count") // 10  # Assuming 1 ASO handles 10 issues
     city_aso_requirements = df.groupby("City")["issues_per_aso"].max().to_dict()
     response += "Estimated ASOs required per city:\n" + "\n".join([f"{k}: {v} ASOs" for k, v in city_aso_requirements.items()])


    
    if "resolved" in query:
        resolved_tickets = df[df["status"].str.lower() == "resolved"]
        response += f"Resolved tickets: {len(resolved_tickets)}\n"
    
    if "predict" in query:
        store_id = ''.join(filter(str.isdigit, query))
        if store_id:
            response += predict_issues(int(store_id))
        else:
            response += "Please enter a valid store ID for prediction."
    
    if "trend" in query or "monthly" in query:
        df["month_year"] = df["created_at"].dt.to_period("M")
        monthly_issues = df.groupby("month_year")["ticket_id"].count()
        response += "Generating trend analysis.\n"
        generate_plot("trend")
    
    if not response:
        response = "Sorry, I couldn't understand your question. Try asking about D-Mart issues, categories, stores, locations, trends, or predictions!"
    
    return response.strip()

# Search Input for Queries
user_query = st.text_input("🔍 Ask your question here:", "", key="user_input")

if user_query:
    response = process_query(user_query)
    st.write("💬 Chatbot:", response)
    
    st.session_state.chat_history.append((user_query, response))
    
    st.subheader("Chat History")
    for q, a in st.session_state.chat_history:
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {a}")
