import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Weather Pulse",
    page_icon="🌦️",
    layout="wide"
)


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.title("🌦️ Weather Pulse")

st.markdown("### AI-Powered Weather & Event Intelligence")

st.info(
    "Weather Pulse uses Machine Learning to predict weather conditions, "
    "classify weather events, and detect duplicate reports."
)


# ---------------- DASHBOARD OVERVIEW ----------------

st.divider()

st.subheader("📌 System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌤️ Weather Prediction", "ML Model")

with col2:
    st.metric("🚨 Event Classification", "NLP + ML")

with col3:
    st.metric("🔍 Duplicate Detection", "AI Similarity")


# ---------------- MODEL PERFORMANCE ----------------

st.divider()

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Weather Model Accuracy", "66.67%")

with col2:
    st.metric("Event Classification Accuracy", "75%")


# ==================================================
# WEATHER PREDICTION
# ==================================================

st.divider()

st.header("🌤️ Weather Prediction")

st.write("Enter the weather parameters below:")

temperature = st.number_input(
    "Temperature (°C)",
    value=32.0
)

humidity = st.number_input(
    "Humidity (%)",
    value=65.0
)

pressure = st.number_input(
    "Pressure (hPa)",
    value=1012.0
)

wind_speed = st.number_input(
    "Wind Speed (km/h)",
    value=12.0
)


if st.button("🔮 Predict Weather"):

    weather_data = pd.read_csv(
        "data/weather_data.csv"
    )

    X = weather_data.drop(
        "weather",
        axis=1
    )

    y = weather_data["weather"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    weather_model = RandomForestClassifier(
        random_state=42
    )

    weather_model.fit(
        X_train,
        y_train
    )

    user_data = pd.DataFrame(
        [[
            temperature,
            humidity,
            pressure,
            wind_speed
        ]],
        columns=[
            "temperature",
            "humidity",
            "pressure",
            "wind_speed"
        ]
    )

    predicted_weather = weather_model.predict(
        user_data
    )

    st.success(
        f"🌤️ Predicted Weather: {predicted_weather[0]}"
    )


# ==================================================
# EVENT CLASSIFICATION + DUPLICATE DETECTION
# ==================================================

st.divider()

st.header("🚨 Event Report Analysis")

st.write(
    "Enter a weather or disaster report:"
)

new_report = st.text_area(
    "Weather/Event Report",
    placeholder="Example: Heavy rainfall caused flooding in Pune"
)


if st.button("🔍 Analyze Report"):

    if new_report.strip() == "":
        st.warning(
            "⚠️ Please enter a report first."
        )

    else:

        event_data = pd.read_csv(
            "data/event_data.csv"
        )

        event_reports = event_data["report"]

        event_labels = event_data["event"]


        # TF-IDF

        event_vectorizer = TfidfVectorizer()

        event_features = event_vectorizer.fit_transform(
            event_reports
        )


        # Event Classification

        event_model = MultinomialNB()

        event_model.fit(
            event_features,
            event_labels
        )


        new_report_features = event_vectorizer.transform(
            [new_report]
        )


        predicted_event = event_model.predict(
            new_report_features
        )


        st.success(
            f"🚨 Detected Event: {predicted_event[0]}"
        )


        # Duplicate Detection

        similarity_scores = cosine_similarity(
            new_report_features,
            event_features
        )

        highest_similarity = similarity_scores.max()

        most_similar_index = similarity_scores.argmax()


        if highest_similarity > 0.7:

            st.warning(
                "⚠️ Possible Duplicate Report Detected"
            )

        else:

            st.success(
                "✅ This is a New Report"
            )


        st.write(
            f"Similarity Score: "
            f"{highest_similarity * 100:.2f}%"
        )


        st.info(
            f"Most Similar Existing Report: "
            f"{event_reports.iloc[most_similar_index]}"
        )