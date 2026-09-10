import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load event dataset
data = pd.read_csv("data/event_data.csv")

print("Existing Reports:")
print(data["report"].head())

# Convert reports into numerical vectors
vectorizer = TfidfVectorizer()

report_vectors = vectorizer.fit_transform(data["report"])

print("\nTF-IDF Vector Shape:")
print(report_vectors.shape)

# New report submitted by a user
new_report = "A powerful cyclone caused damage to buildings near the coast"

print("\nNew Report:")
print(new_report)

# Convert new report into a numerical vector
new_report_vector = vectorizer.transform([new_report])

print("\nNew Report Vector Shape:")
print(new_report_vector.shape)

# Compare new report with all existing reports
similarity_scores = cosine_similarity(
    new_report_vector,
    report_vectors
)

print("\nSimilarity Scores:")
print(similarity_scores)

# Find highest similarity score
max_similarity = similarity_scores.max()

print("\nHighest Similarity Score:")
print(max_similarity)

# Set duplicate threshold
threshold = 0.7

if max_similarity >= threshold:
    print("\n⚠️ Possible Duplicate Report Detected!")
else:
    print("\n✅ This is a New Report.")


    # Find the index of the most similar report
most_similar_index = similarity_scores.argmax()

# Get the matched report
matched_report = data.iloc[most_similar_index]["report"]

print("\nMost Similar Existing Report:")
print(matched_report)
