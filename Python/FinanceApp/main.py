import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os


# Streamlit App Setup

st.set_page_config(page_title="Financial App", page_icon="💸", layout="wide")

# File where categories are stored for persistence
category_file = "categories.json"

# Initialize default categories in session state
if "categories" not in st.session_state:
    st.session_state.categories = {
        "Uncategorized": []
    }

# Load categories from JSON file if it exists
if os.path.exists(category_file):
    with open(category_file, "r") as f:
        st.session_state.categories = json.load(f)

# Save category dictionary back to JSON file
def save_categories():
    with open(category_file, "w") as f:
        json.dump(st.session_state.categories, f)



# Auto-categorize transactions by keywords
def categorize_transactions(df):
    # Default all rows to the 'Uncategorized' category
    df["Category"] = "Uncategorized"
    
    # Loop through stored categories + keywords
    for category, keywords in st.session_state.categories.items():
        
        # Skip empty categories or the Uncategorized 
        if category == "Uncategorized" or not keywords:
            continue
        
        # Prepare keyword list for case-insensitive matching
        lowered_keywords = [keyword.lower().strip() for keyword in keywords]
        
        # Compare each transaction's "Details" field against keywords
        for idx, row in df.iterrows():
            details = row["Details"].lower().strip()
            if details in lowered_keywords:
                df.at[idx, "Category"] = category
    
    return df


# Load and clean uploaded transaction CSV
def load_transactions(file):
    try:
        df = pd.read_csv(file)
        
        # Remove all empty/unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Clean column names
        df.columns = [col.strip() for col in df.columns]

        # Convert Amount column to float
        df["Amount"] = df["Amount"].str.replace(",", "").astype(float)

        # Convert Date column from string to datetime
        df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
        
        # Apply automatic categorization logic
        return categorize_transactions(df)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None



# Add a keyword to a category & persist it
def add_keyword_to_category(category, keyword):
    keyword = keyword.strip()

    # Only add if keyword is not empty and doesn't already exist
    if keyword and keyword not in st.session_state.categories[category]:
        st.session_state.categories[category].append(keyword)
        save_categories()
        return True
    return False



# Main Streamlit App
def main():
    st.title("Simple Financial Dashboard")

    # CSV upload widget
    uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = load_transactions(uploaded_file)
        
        if df is not None:
            # Split data into debits (expenses) and credits (income/payments)
            debits_df = df[df["Debit/Credit"] == "Debit"].copy()
            credits_df = df[df["Debit/Credit"] == "Credit"].copy()
            
            # Make a copy where user edits won't break original
            st.session_state.debits_df = debits_df.copy()
            
            # Two-tab layout: Expenses + Payments
            tab1, tab2 = st.tabs(["Expenses (Debits)", "Payments (Credits)"])
            
            # EXPENSES TAB
            with tab1:
                # Add new category UI input
                new_category = st.text_input("New Category Name")
                add_button = st.button("Add Category")
                
                # Add category to list and save to JSON file
                if add_button and new_category:
                    if new_category not in st.session_state.categories:
                        st.session_state.categories[new_category] = []
                        save_categories()
                        st.rerun()  # Refresh app to show updated categories
                
                st.subheader("Your Expenses")

                # Editable table for category assignment
                edited_df = st.data_editor(
                    st.session_state.debits_df[["Date", "Details", "Amount", "Category"]],
                    column_config={
                        "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                        "Amount": st.column_config.NumberColumn("Amount", format="%.2f AED"),
                        "Category": st.column_config.SelectboxColumn(
                            "Category",
                            options=list(st.session_state.categories.keys())
                        )
                    },
                    hide_index=True,
                    use_container_width=True,
                    key="category_editor"
                )
                
                # Save category changes
                save_button = st.button("Apply Changes", type="primary")
                if save_button:
                    for idx, row in edited_df.iterrows():
                        new_category = row["Category"]

                        # Skip if unchanged
                        if new_category == st.session_state.debits_df.at[idx, "Category"]:
                            continue
                        
                        details = row["Details"]

                        # Update category assignment
                        st.session_state.debits_df.at[idx, "Category"] = new_category

                        # Learn this keyword for future auto-categorizing
                        add_keyword_to_category(new_category, details)
                
                # Expense totals by category
                st.subheader('Expense Summary')
                category_totals = st.session_state.debits_df.groupby("Category")["Amount"].sum().reset_index()
                category_totals = category_totals.sort_values("Amount", ascending=False)
                
                # Show summary table
                st.dataframe(
                    category_totals,
                    column_config={"Amount": st.column_config.NumberColumn("Amount", format="%.2f AED")},
                    use_container_width=True,
                    hide_index=True
                )
                
                # Pie chart visualization
                fig = px.pie(
                    category_totals,
                    values="Amount",
                    names="Category",
                    title="Expenses by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
                        
            
            # PAYMENTS TAB
            with tab2:
                st.subheader("Payments Summary")
                total_payments = credits_df["Amount"].sum()

                # Summary metric for total incoming payments
                st.metric("Total Payments", f"{total_payments:.2f} AED")

                # Display raw credit transactions
                st.write(credits_df)
        
        
# Run the app
main()
