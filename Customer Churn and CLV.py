# Objective: Analyze customer churn patterns predict customer lifetime values (CLV model),
# and develop strategies to reduce churn in a health insurance company's customer base.
# Data source: health insurance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
from lifetimes.utils import summary_data_from_transaction_data

# Load and preprocess the customer churn and transaction data
churn_data = pd.read_csv('customer_churn_data.csv')
transaction_data = pd.read_csv('customer_transaction_data.csv')

# Data preprocessing and feature engineering
churn_data['BMI'] = churn_data['Weight'] / (churn_data['Height'] / 100) ** 2
churn_data['LogIncome'] = np.log(churn_data['Income'] + 1)
churn_data['IsSmoker'] = (churn_data['Smoker'] == 'Yes').astype(int)

# Create a summary dataset for CLV modeling (Customer Lifetime Value)
summary_data = summary_data_from_transaction_data(transaction_data, 'Customer_ID', 'Transaction_Date')

# Split the data into training and testing sets
X = churn_data[['Age', 'BMI', 'LogIncome', 'IsSmoker']]
y = churn_data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a customer churn prediction model (Random Forest)
churn_model = RandomForestClassifier()
churn_model.fit(X_train, y_train)

# Make predictions and evaluate the churn model
churn_predictions = churn_model.predict(X_test)
churn_accuracy = accuracy_score(y_test, churn_predictions)
churn_conf_matrix = confusion_matrix(y_test, churn_predictions)

print(f'Churn Model Accuracy: {churn_accuracy}')
print('Churn Model Confusion Matrix:')
print(churn_conf_matrix)
print(classification_report(y_test, churn_predictions))

# Customer Lifetime Value (CLV) modeling using Beta Geo Fitter (Not ParetaNBDFitter).
# (Note: Assume Beta-Geometric distribution for Beta Geo Fitter)
# Suitable for modeling customer transactions in scenarios where customers make repeat purchases
# and eventually become "inactive" or "churn."

bgf = BetaGeoFitter(penalizer_coef=0.01)
# Regularization is applied which can dealing with noisy data or preventing extreme paremter values.

bgf.fit(
    summary_data['frequency'],
    summary_data['recency'],
    summary_data['T'])

# Visualize frequency-recency matrix and probability alive matrix
plot_frequency_recency_matrix(bgf)
plt.title('Frequency-Recency Matrix')
plt.show()

plot_probability_alive_matrix(bgf)
plt.title('Probability Alive Matrix')
plt.show()

# Predict future customer transactions and CLV
t = 10  # Predict CLV for the next 10 periods
summary_data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
                                      t,
                                      summary_data['frequency'],
                                      summary_data['recency'],
                                      summary_data['T'])

summary_data['predicted_clv'] = bgf.customer_lifetime_value(
    bgf,
    summary_data['frequency'],
    summary_data['recency'],
    summary_data['T'],
    summary_data['predicted_purchases'],
    time=t,
    discount_rate=0.1  # Adjust the discount rate as needed
)

# Visualize predicted CLV distribution
plt.hist(summary_data['predicted_clv'], bins=30)
plt.xlabel('Predicted CLV')
plt.ylabel('Count')
plt.title('Predicted CLV Distribution')
plt.show()
