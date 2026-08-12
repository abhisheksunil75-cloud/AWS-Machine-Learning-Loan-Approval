import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

np.random.seed(42)
num_samples = 1000

income = np.random.randint(25000, 150000, num_samples)
credit_score = np.random.randint(550, 850, num_samples)
ev_price = np.random.randint(15000, 60000, num_samples)
down_payment = np.random.randint(2000, 15000, num_samples)
loan_term = np.random.choice([24, 36, 48, 60, 72], num_samples)

approved = ((credit_score > 640) & (down_payment >= 0.10 * ev_price)).astype(int)

X = np.column_stack((income, credit_score, ev_price, down_payment, loan_term))
y = approved

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, 'model.joblib')
print("? model.joblib trained and saved successfully.")
