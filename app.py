import streamlit as st
import pandas as pd
from src import data_processing, nlp_categorization, expense_prediction, suggestion_engine, visualize

st.title("AI-Powered Personal Finance Advisor")

uploaded_file = st.file_uploader("Upload your transactions CSV", type=["csv"])
if uploaded_file:
    df = data_processing.load_data(uploaded_file)
    expenses = data_processing.preprocess_data(df)

    # Train or load categorizer
    try:
        categorizer = nlp_categorization.load_categorizer()
    except:
        categorizer = nlp_categorization.train_categorizer()

    # Categorize transactions
    expenses['category'] = nlp_categorization.categorize_transactions(expenses['description'], categorizer)

    st.subheader("Categorized Transactions")
    st.dataframe(expenses)

    # Visualize expenses
    st.subheader("Expense Distribution")
    visualize.plot_expense_pie(expenses)

    # Prepare data and train prediction model
    monthly_expense = expense_prediction.prepare_monthly_expense(expenses)
    model = expense_prediction.train_expense_model(monthly_expense)

    last_month_num = monthly_expense['month_num'].max()
    prediction = expense_prediction.predict_next_month(model, last_month_num)

    st.subheader("Expense Prediction")
    st.write(f"Predicted expenses for next month: **${prediction:.2f}**")

    # Suggestions
    st.subheader("Personalized Suggestions")
    suggestions = suggestion_engine.generate_suggestions(expenses)
    for s in suggestions:
        st.write(f"- {s}")
