import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

print("\n========================================")
print("        WEATHER PULSE - SIH SYSTEM")
print("========================================")
print("AI-Powered Weather & Event Analysis")
print("========================================")




# Load weather dataset
weather_data = pd.read_csv("data/weather_data.csv")

print("\nWeather Dataset Loaded Successfully!")

# Separate input features and target
X = weather_data.drop("weather", axis=1)
y = weather_data["weather"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the weather prediction model
weather_model = RandomForestClassifier(random_state=42)

weather_model.fit(X_train, y_train)

print("\nWeather Prediction Model Trained Successfully!")

# Test the weather prediction model
weather_predictions = weather_model.predict(X_test)

weather_accuracy = accuracy_score(y_test, weather_predictions)

print("Weather Model Accuracy:", weather_accuracy * 100, "%")

print("\n--- WEATHER PREDICTION ---")

temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
pressure = float(input("Enter Pressure: "))
wind_speed = float(input("Enter Wind Speed: "))

# Create input data for prediction
user_data = pd.DataFrame(
    [[temperature, humidity, pressure, wind_speed]],
    columns=["temperature", "humidity", "pressure", "wind_speed"]
)

# Predict weather
predicted_weather = weather_model.predict(user_data)

print("\nPredicted Weather:", predicted_weather[0])

# Load event dataset
event_data = pd.read_csv("data/event_data.csv")

print("\nEvent Dataset Loaded Successfully!")

# Separate event reports and labels
event_reports = event_data["report"]
event_labels = event_data["event"]

# Convert text reports into numerical features
event_vectorizer = TfidfVectorizer()

event_features = event_vectorizer.fit_transform(event_reports)

print("\nEvent TF-IDF Shape:", event_features.shape)

# Split event data into training and testing sets
event_X_train, event_X_test, event_y_train, event_y_test = train_test_split(
    event_features,
    event_labels,
    test_size=0.2,
    random_state=42
)

# Create and train event classification model
event_model = MultinomialNB()

event_model.fit(event_X_train, event_y_train)

print("\nEvent Classification Model Trained Successfully!")

# Test event classification model
event_predictions = event_model.predict(event_X_test)

event_accuracy = accuracy_score(event_y_test, event_predictions)

print("Event Model Accuracy:", event_accuracy * 100, "%")

print("\n--- EVENT CLASSIFICATION ---")

new_report = input("Enter a weather/event report: ")

# Convert user report into TF-IDF features
new_report_features = event_vectorizer.transform([new_report])

# Predict the event
predicted_event = event_model.predict(new_report_features)

print("\nPredicted Event:", predicted_event[0])

# Duplicate Detection
existing_reports = event_data["report"]

print("\n--- DUPLICATE DETECTION ---")
print("Checking if this report already exists...")

# Calculate similarity between new report and existing reports
similarity_scores = cosine_similarity(
    new_report_features,
    event_features
)

highest_similarity = similarity_scores.max()

print("Highest Similarity Score:", highest_similarity)

# Check if report is duplicate
if highest_similarity > 0.7:
    print("⚠️ Possible Duplicate Report Detected!")
else:
    print("✅ This is a New Report.")


# Find the most similar existing report
most_similar_index = similarity_scores.argmax()

print("\nMost Similar Existing Report:")
print(existing_reports.iloc[most_similar_index])

