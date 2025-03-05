import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

# Load NLP model for query understanding
nlp = spacy.load("en_core_web_sm")

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
st.write("Ask anything about D-Mart issues, terminals, trends, categories, predictions, or analysis!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_active_terminals(df_filtered):
    """
    Returns the correct count of active terminals.
    Active terminal = NOT in churned list or still has transactions.
    """
    print(f"Total Terminals in Dataset: {df_filtered['cf_utid'].nunique()}")

    # Identifying churned terminals
    churned_terminals = df_filtered[
        (df_filtered["issue_type"].str.lower() == "Churn Request") & 
        (df_filtered["status"].str.lower() == "closed")
    ]["cf_utid"].unique()

    print(f"Total Churned Terminals: {len(churned_terminals)}")

    # Active terminals = Exclude churned ones
    active_terminals = df_filtered[~df_filtered["cf_utid"].isin(churned_terminals)]["cf_utid"].nunique()

    print(f"Active Terminals (Corrected): {active_terminals}")
    return active_terminals


def process_query(query):
    query = query.lower()
    doc = nlp(query)
    response = ""

    # **Step 1: Extract Date (Month-Year)**
    date_match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december) \d{4}", query)
    month_year = date_match.group() if date_match else df["created_at"].max().strftime("%B %Y")

    # **Step 2: Remove Duplicates Before Processing**
    df.drop_duplicates(subset=["cf_utid"], keep="last", inplace=True)

    # **Step 3: Filter Data for Correct Month-Year**
    df_filtered = df[df["created_at"].dt.strftime("%B %Y").str.lower() == month_year.lower()]
    
    # **DEBUG: Print Data Filtering Results**
    print(f"📌 Query Month-Year: {month_year}")
    print(f"📌 Total Rows in df_filtered (after duplicates removed): {len(df_filtered)}")

    # ✅ **Terminal Queries**
    if "terminal" in query:
        # **Step 4: Identify Churned Terminals**
        churned_terminals = df_filtered[
            (df_filtered["issue_type"].str.lower() == "churn utid") &
            (df_filtered["status"].str.lower() == "closed")
        ]["cf_utid"].unique()

        # **Step 5: Identify Active Terminals**
        total_terminals = df_filtered["cf_utid"].nunique()
        active_terminals = total_terminals - len(churned_terminals)

        # **DEBUG: Print Terminal Counts**
        # print(f"📌 Total Terminals in Dataset: {df['cf_utid'].nunique()}")
        # print(f"📌 Total Churned Terminals: {len(churned_terminals)}")
        # print(f"📌 Active Terminals: {active_terminals}")

        response += f"✅ Active terminals as of {month_year}: {active_terminals}\n"

    # ✅ **Store Queries**
    if "store" in query:
        store_churn_status = df_filtered[df_filtered["cf_utid"].isin(churned_terminals)].groupby("Store")["cf_utid"].nunique()
        active_stores = df_filtered[~df_filtered["Store"].isin(store_churn_status.index)]["Store"].nunique()

        # **DEBUG: Print Store Counts**
        print(f"📌 Active Stores: {active_stores}")

        response += f"🏬 Active stores as of {month_year}: {active_stores}\n"

    # ✅ **Final Check**
    if not response:
        response = "❌ Sorry, I couldn't understand your question. Try asking about terminals, stores, transaction dates, trends, or predictions!"
    
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
