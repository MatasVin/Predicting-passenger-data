# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("Ruter_data.csv", sep=";")

# Convert Dato to datetime and extract day, month, year
data['Dato'] = pd.to_datetime(data['Dato'], dayfirst=True)
data['Year'] = data['Dato'].dt.year
data['Month'] = data['Dato'].dt.month
data['Day'] = data['Dato'].dt.day

# Remove outliers based on IQR for 'Passasjerer_Ombord'
Q1 = data['Passasjerer_Ombord'].quantile(0.25)
Q3 = data['Passasjerer_Ombord'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_data = data[(data['Passasjerer_Ombord'] >= lower_bound) & (data['Passasjerer_Ombord'] <= upper_bound)]

# Select a specific bus for prediction, for this example let's choose "Linjenavn" value '100'
bus_data = filtered_data[filtered_data['Linjenavn'] == '100']

# Features and target
X = bus_data[['Year', 'Month', 'Day']]
y = bus_data['Passasjerer_Ombord']

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

def predict_passengers(date):
    date_obj = pd.to_datetime(date, dayfirst=True)
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    predicted_value = model.predict([[year, month, day]])
    return int(predicted_value[0])

# Test the prediction function
input_date = "2022-07-12"
predicted_passengers = predict_passengers(input_date)
print(f"Predicted number of passengers on {input_date}: {predicted_passengers}")
