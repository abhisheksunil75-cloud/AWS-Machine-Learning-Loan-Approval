import streamlit as st
import requests

# Configure the Streamlit page title and layout
st.set_page_config(page_title="EV Loan Portal", layout="centered")

# Display the main title and description
st.title("⚡ EV Loan Approval Portal")
st.write("Submit applicant details to trigger the live AWS MLOps pipeline.")

# AWS API Gateway endpoint
API_URL = "https://f5drt44e57.execute-api.ap-south-1.amazonaws.com/predict"

# Create two columns for arranging the input fields
col1, col2 = st.columns(2)

# First column: applicant financial details
with col1:
    # Input field for the requested loan amount
    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=10000,
        max_value=2000000,
        value=540000,
        step=10000
    )

    # Slider for entering the applicant's CIBIL score
    cibil_score = st.slider(
        "CIBIL Score",
        min_value=300,
        max_value=900,
        value=750
    )

    # Input field for the applicant's annual income
    annual_income = st.number_input(
        "Annual Income (₹)",
        min_value=50000,
        max_value=5000000,
        value=140000,
        step=10000
    )

# Second column: existing financial commitments and loan tenure
with col2:
    # Input field for the applicant's existing monthly EMI
    existing_emi = st.number_input(
        "Existing Monthly EMI (₹)",
        min_value=0,
        max_value=200000,
        value=15000,
        step=1000
    )

    # Dropdown menu for selecting the loan repayment period
    tenure_months = st.selectbox(
        "Tenure (Months)",
        options=[12, 24, 36, 48, 60],
        index=2
    )

# Execute the following code when the user clicks the button
if st.button("Submit Application", type="primary"):

    # Create the request payload expected by the AWS API
    # Float conversion ensures that all model input values
    # are sent as numeric floating-point values
    payload = {
        "inputs": [[
            float(loan_amount),
            float(cibil_score),
            float(annual_income),
            float(existing_emi),
            float(tenure_months)
        ]]
    }

    # Display a loading message while the API request is being processed
    with st.spinner("Calling API Gateway -> Lambda -> SageMaker..."):
        try:
            # Send the applicant data to the AWS API using a POST request
            res = requests.post(
                API_URL,
                json=payload,
                timeout=15
            )

            # Check whether the API request was successful
            if res.status_code == 200:

                # Convert the API response from JSON into a Python dictionary
                data = res.json()

                # Get the processing status from the response
                # Use "PROCESSED" if no status is returned
                status = data.get("status", "PROCESSED")

                # Check the machine-learning prediction
                # Prediction 1 means the loan is approved
                if data.get("prediction") == 1:
                    st.success(
                        f"🎉 Decision: {status} (Approved)"
                    )

                # Any prediction other than 1 is treated as rejected
                else:
                    st.error(
                        f"❌ Decision: {status} (Rejected)"
                    )

                # Inform the user that an email notification
                # was sent through Amazon SNS
                st.info("📩 Email alert dispatched via Amazon SNS.")

            # Handle API errors such as 400, 403, 500, etc.
            else:
                st.error(
                    f"Server Error {res.status_code}: {res.text}"
                )

        # Handle connection errors, timeout errors,
        # and other unexpected exceptions
        except Exception as e:
            st.error(
                f"Failed to connect to endpoint: {e}"
            )
