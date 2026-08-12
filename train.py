# Import NumPy for generating and working with numerical data
import numpy as np

# Import Random Forest Classifier from scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import joblib to save the trained machine learning model
import joblib


# Set a random seed so that the randomly generated data
# remains the same every time the code is executed
np.random.seed(42)


# Define the number of samples in our dataset
num_samples = 1000


# Generate random customer income values
# Values range from ₹25,000 to ₹149,999
income = np.random.randint(25000, 150000, num_samples)


# Generate random credit scores
# Values range from 550 to 849
credit_score = np.random.randint(550, 850, num_samples)


# Generate random EV prices
# Values range from ₹15,000 to ₹59,999
ev_price = np.random.randint(15000, 60000, num_samples)


# Generate random down-payment amounts
# Values range from ₹2,000 to ₹14,999
down_payment = np.random.randint(2000, 15000, num_samples)


# Randomly assign loan terms from the available options
# 24, 36, 48, 60, or 72 months
loan_term = np.random.choice([24, 36, 48, 60, 72], num_samples)


# Create the target variable (loan approval)
#
# Loan is approved (1) when:
# 1. Credit score is greater than 640
# 2. Down payment is at least 10% of the EV price
#
# Otherwise, the loan is rejected (0)
approved = (
    (credit_score > 640) &
    (down_payment >= 0.10 * ev_price)
).astype(int)


# Combine all input features into a single NumPy array
#
# Each row represents one customer
# Columns represent:
# income, credit_score, ev_price, down_payment, loan_term
X = np.column_stack((
    income,
    credit_score,
    ev_price,
    down_payment,
    loan_term
))


# Store the target/output variable
# y contains either:
# 1 = Loan Approved
# 0 = Loan Rejected
y = approved


# Split the dataset into training and testing data
#
# test_size=0.2 means:
# 80% of the data → training
# 20% of the data → testing
#
# random_state=42 ensures the same split every time
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create a Random Forest classification model
#
# n_estimators=100 means the Random Forest
# will contain 100 decision trees
#
# random_state=42 makes the model reproducible
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the Random Forest model using the training data
clf.fit(X_train, y_train)


# Save the trained model to a file
#
# The model can later be loaded without training it again
joblib.dump(clf, 'model.joblib')


# Display a confirmation message
print("Model model.joblib trained and saved successfully.")
