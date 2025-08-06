import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

@st.cache_data
def load_model_and_columns():
    model = joblib.load('aml_xgb_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

model, model_columns = load_model_and_columns()


# Defining the desired JSON output structure with Pydantic
class Verdict(BaseModel):
    agreed_with_model: bool = Field(description="Whether your final verdict agrees with the initial model's prediction.")
    final_verdict: str = Field(description="Your final assessment, must be either 'High Risk' or 'Low Risk'.")
    reasoning: str = Field(description="A brief, one-sentence explanation for your decision.")

# Creating a LangChain function to get the "second opinion"
def get_llm_verdict(input_data, xgb_prediction, xgb_probability):
    """
    Uses a LangChain chain to get a final verdict from Gemini.
    """
    try:
        # Initializing the LangChain chat model
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                     google_api_key=st.secrets["GEMINI_API_KEY"],
                                     temperature=0)     # Low temperature for consistent, structured output

        # Creating a JSON parser with instructions from our Pydantic model
        parser = JsonOutputParser(pydantic_object=Verdict)

        # Converting prediction to human-readable text
        xgb_verdict_text = "High Risk" if xgb_prediction[0] == 1 else "Low Risk"
        input_json = input_data.to_json(orient='records')

        # Creating the prompt template
        prompt_template = """
        You are an expert financial crime analyst providing a second opinion.

        A machine learning model has analyzed a transaction with the following details:
        {input_json}

        The model's initial prediction is: '{xgb_verdict}' with a confidence score of {xgb_prob}.

        Your task is to analyze this transaction based on these established money laundering red flags:
        1. Large transactions from accounts with little or no history are highly suspicious.
        2. Transactions involving unnecessary complexity (e.g., cross-border transfers without currency exchange) are red flags.
        3. Cash-equivalent payment types like 'Wire' for large sums are higher risk.
        4. Accounts acting as simple passthroughs (high fan-in and fan-out, low history) are suspicious.

        Based on these rules, do you agree with the model's prediction?

        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Creating the chain by piping components together
        chain = prompt | llm | parser

        # Invoking the chain with the transaction data
        llm_result = chain.invoke({
            "input_json": input_json,
            "xgb_verdict": xgb_verdict_text,
            "xgb_prob": f"{xgb_probability:.2%}"
        })
        return llm_result

    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        return None

st.title("Anti Money Laundering (AML) Transaction Detector with AI-based Verification")
st.write("Enter transaction details to get a money laundering risk assessment from our XGBoost model, verified by a secondary AI analyst.")

st.sidebar.header("Scenario Details")
def user_input_features():
    st.sidebar.subheader("Basic Transaction Info")
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, format="%.2f")
    payment_type = st.sidebar.selectbox("Payment Type", ('Wire', 'Credit Card', 'Debit Card', 'ACH', 'Cash'))
    payment_currency = st.sidebar.selectbox("Payment Currency", ('USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD'))
    received_currency = st.sidebar.selectbox("Received Currency", ('USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD'))
    sender_bank_loc = st.sidebar.selectbox("Sender Bank Location", ('US', 'DE', 'GB', 'AU', 'CA', 'JP'))
    receiver_bank_loc = st.sidebar.selectbox("Receiver Bank Location", ('US', 'DE', 'GB', 'AU', 'CA', 'JP'))
    st.sidebar.subheader("Historical Account Behavior")
    sender_txn_count = st.sidebar.number_input("Sender's Past Transaction Count", min_value=0, step=1)
    sender_total_amt = st.sidebar.number_input("Sender's Total Past Transaction Amount", min_value=0.0, format="%.2f")
    receiver_txn_count = st.sidebar.number_input("Receiver's Past Transaction Count", min_value=0, step=1)
    receiver_total_amt = st.sidebar.number_input("Receiver's Total Past Transaction Amount", min_value=0.0, format="%.2f")
    sender_fan_out = st.sidebar.number_input("# of Unique Accounts Sender Has Paid", min_value=0, step=1)
    receiver_fan_in = st.sidebar.number_input("# of Unique Accounts Receiver Was Paid By", min_value=0, step=1)
    data = {'Amount': amount, 'Payment_type': payment_type, 'Payment_currency': payment_currency, 'Received_currency': received_currency, 'Sender_bank_location': sender_bank_loc, 'Receiver_bank_location': receiver_bank_loc, 'sender_transaction_count': sender_txn_count, 'sender_total_amount': sender_total_amt, 'receiver_transaction_count': receiver_txn_count, 'receiver_total_amount': receiver_total_amt, 'sender_fan_out': sender_fan_out, 'receiver_fan_in': receiver_fan_in}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

st.subheader("Scenario Being Tested")
st.write(input_df)

proc_df = input_df.copy()
proc_df['log_amount'] = np.log1p(proc_df['Amount'])
proc_df['is_foreign_exchange'] = (proc_df['Payment_currency'] != proc_df['Received_currency']).astype(int)
proc_df['is_cross_border'] = (proc_df['Sender_bank_location'] != proc_df['Receiver_bank_location']).astype(int)
proc_df['hour_of_day'] = pd.Timestamp.now().hour
proc_df['day_of_week'] = pd.Timestamp.now().dayofweek
proc_df_encoded = pd.get_dummies(proc_df)
proc_df_aligned = proc_df_encoded.reindex(columns=model_columns, fill_value=0)


if st.button("XGBoost Prediction -- Analyze with AI Validation"):
    xgb_prediction = model.predict(proc_df_aligned)
    xgb_prediction_proba = model.predict_proba(proc_df_aligned)[0][1]

    with st.spinner("Analyzing with XGBoost and getting a second opinion from our AI Analyst..."):
        llm_result = get_llm_verdict(input_df, xgb_prediction, xgb_prediction_proba)

    if llm_result:
        st.subheader("Final Verdict")
        if llm_result['final_verdict'] == 'High Risk':
            st.error(f"**Verdict:** {llm_result['final_verdict']}")
        else:
            st.success(f"**Verdict:** {llm_result['final_verdict']}")
        
        st.info(f"**AI Analyst's Reasoning:** {llm_result['reasoning']}")

        # if not llm_result['agreed_with_model']:
        #     st.warning("Note: The AI Analyst's final verdict overrode the initial model's prediction")