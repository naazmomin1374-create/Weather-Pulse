# 🌦️ Weather Pulse

### AI-Powered Weather & Event Intelligence System

Weather Pulse is a Machine Learning and NLP-based system designed to analyze weather information and weather-related reports.

The project combines **weather prediction, weather event classification, and duplicate report detection** in one simple dashboard.

---

## 🚀 Features

### 🌤️ Weather Prediction
Uses a **Random Forest Classifier** to predict weather conditions based on:

- Temperature
- Humidity
- Atmospheric Pressure
- Wind Speed

### 🚨 Weather Event Classification
Uses **Natural Language Processing (NLP)** and **TF-IDF** with a **Multinomial Naive Bayes** model to classify weather-related reports into events such as:

- Flood
- Heatwave
- Fog
- Thunderstorm

### 🔍 Duplicate Report Detection
Uses **Cosine Similarity** to compare a new weather report with existing reports and identify whether it may be a duplicate.

The system also displays the most similar existing report and its similarity score.

---

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Natural Language Processing (NLP)
- TF-IDF
- Random Forest
- Multinomial Naive Bayes
- Cosine Similarity

---

## 📊 Model Performance

| Module | Model / Technique | Accuracy |
|---|---|---:|
| Weather Prediction | Random Forest Classifier | 66.67% |
| Event Classification | TF-IDF + Multinomial Naive Bayes | 75% |
| Duplicate Detection | Cosine Similarity | Similarity-based |

> Accuracy values are based on the current project dataset and may change with a larger or different dataset.

---

## 📁 Project Structure

```text
Weather-Pulse/
│
├── app.py
├── dashboard.py
├── README.md
│
└── data/
    ├── duplicate_detector.py
    ├── event_classifier.py
    ├── event_data.csv
    ├── main.py
    └── weather_data.csv
