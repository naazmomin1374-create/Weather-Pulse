# 🌦️ Weather Pulse

### AI-Powered Weather & Event Intelligence

Weather Pulse is a Machine Learning-based system designed to analyze weather data, classify weather-related events, and detect duplicate weather reports.

## 🚀 Features

- 🌤️ Weather condition prediction using Machine Learning
- 🚨 Weather event classification using NLP
- 🔍 Duplicate report detection using text similarity
- 📊 Interactive Streamlit dashboard
- 📁 CSV-based datasets
- 🤖 Machine Learning models using Python and Scikit-learn

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF
- Multinomial Naive Bayes
- Random Forest Classifier
- Cosine Similarity

## 📊 Model Performance

| Module | Technique | Accuracy / Method |
|---|---|---|
| Weather Prediction | Random Forest Classifier | 66.67% |
| Event Classification | TF-IDF + Multinomial Naive Bayes | 75% |
| Duplicate Detection | Cosine Similarity | Similarity-based |

> Accuracy values are based on the current project dataset and may change with a larger or different dataset.

## 🔍 How Duplicate Detection Works

The system compares a new weather/event report with existing reports using **Cosine Similarity**.

- High similarity → Possible duplicate report
- Low similarity → New report

For example, an exact report already present in the dataset can produce a **100% similarity score**. A new report can have a lower score, such as **51.31%**, and be identified as a new report.

## 📂 Project Structure

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
▶️ How to Run
1. Clone the repository
git clone https://github.com/naazmomin1374-create/Weather-Pulse.git
2. Open the project folder
cd Weather-Pulse
3. Install the required libraries
pip install pandas scikit-learn streamlit
4. Run the dashboard
python -m streamlit run dashboard.py

The Weather Pulse dashboard will open in your browser.

💡 Project Purpose

Weather Pulse demonstrates how Machine Learning and Natural Language Processing can be used to support weather and disaster-related data analysis.

The system combines:

Weather Prediction + Event Classification + Duplicate Detection

into one interactive dashboard.

👩‍💻 Project

Weather Pulse — Smart India Hackathon Project

Developed using Python, Machine Learning, NLP, and Streamlit.
