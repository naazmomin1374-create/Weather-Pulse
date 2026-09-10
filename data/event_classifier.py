import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load event dataset
data = pd.read_csv("data/event_data.csv")

print("Event Dataset:")
print(data.head())

# Convert all reports to lowercase
data["clean_report"] = data["report"].str.lower()

print("\nCleaned Dataset:")
print(data[["report", "clean_report"]].head())

# Remove punctuation
data["clean_report"] = data["clean_report"].apply(
    lambda text: re.sub(r"[^\w\s]", "", text)
)

print("\nAfter removing punctuation:")
print(data[["clean_report"]].head())

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert cleaned text into numbers
X = vectorizer.fit_transform(data["clean_report"])

print("\nTF-IDF Shape:")
print(X.shape)

# Target labels
y = data["event"]

print("\nEvent Labels:")
print(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# Create the classification model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Predict events for testing data
predictions = model.predict(X_test)

print("\nPredicted Events:")
print(predictions)

print("\nActual Events:")
print(y_test.values)

# Calculate model accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy * 100, "%")
