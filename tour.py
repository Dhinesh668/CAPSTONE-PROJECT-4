import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config with title and icon
st.set_page_config(page_title="Tourism Visit Mode Predictor", page_icon="🌍", layout="centered")

# Title bar
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌍 Tourism Visit Mode Prediction App</h1>", unsafe_allow_html=True)
st.write("---")

@st.cache_data
def load_data():
    return pd.read_excel("Test8.xlsx")

@st.cache_resource
def load_model():
    return joblib.load("classification.pkl")

df = load_data()
model = load_model()

# Debug: show columns in dataframe to verify all needed features are present
st.write("Columns in DataFrame:", df.columns.tolist())

# User preferences on main page (no sidebar)
st.header("User Preferences")

continent = st.selectbox("Select Continent", df['Continent'].unique())
country = st.selectbox(
    "Select Country",
    df[df['Continent'] == continent]['Country'].unique()
)
city = st.selectbox(
    "Select City",
    df[(df['Continent'] == continent) & (df['Country'] == country)]['City'].unique()
)

st.write("---")

# Display chosen inputs
st.write("### 🧭 Selected Location & Preference")
st.write(f"**Continent:** {continent}")
st.write(f"**Country:** {country}")
st.write(f"**City:** {city}")

# Predict button triggers prediction
if st.button("Predict Visit Mode"):
    try:
        sample = df[
            (df['Continent'] == continent) &
            (df['Country'] == country) &
            (df['City'] == city)
        ].iloc[0]

        # Calculate user features
        user_ratings = df[df['UserId'] == sample['UserId']]['Rating']
        user_avg_rating = user_ratings.mean() if not user_ratings.empty else 0
        user_rating_count = len(user_ratings)

        # Calculate attraction features
        attraction_ratings = df[df['AttractionId'] == sample['AttractionId']]['Rating']
        attraction_avg_rating = attraction_ratings.mean() if not attraction_ratings.empty else 0
        attraction_visit_count = len(attraction_ratings)

        # Calculate city features
        city_ratings = df[df['City'] == sample['City']]['Rating']
        city_avg_rating = city_ratings.mean() if not city_ratings.empty else 0
        city_visit_count = len(city_ratings)

        # VisitMonth related features - adjust the column name if different
        visit_month_column = 'VisitMonth' if 'VisitMonth' in df.columns else 'Visit_YearMonth'  # fallback example
        visit_month_val = sample.get(visit_month_column, 1)  # fallback to 1 if missing

        visit_month_sin = np.sin(2 * np.pi * visit_month_val / 12)
        visit_month_cos = np.cos(2 * np.pi * visit_month_val / 12)

        # Construct feature vector matching your model features
        your_features = [
            sample['ContinentId'],
            sample['RegionId'],
            sample['CountryId'],
            sample.get('CityId', 0),
            sample.get('AttractionId', 0),
            sample['AttractionTypeId'],
            sample.get('Visit_YearMonth', 0),
            user_avg_rating,
            user_rating_count,
            attraction_avg_rating,
            attraction_visit_count,
            city_avg_rating,
            city_visit_count,
            sample.get('user_total_visits', 0),
            sample.get('user_preferred_attraction_type', 0),
            visit_month_sin,
            visit_month_cos,
            sample.get('attraction_visit_ratio', 0),
            sample.get('month_ratio', 0),
            sample.get('month_city', 0),
            sample.get('rating_ratio', 0),
            sample.get('Attraction_Avg_Rating_Hist', 0),
            sample.get('User_Type_Avg_Rating_Hist', 0)
        ]

        predicted_mode = model.predict([your_features])[0]
        st.success(f"### 🔮 Predicted Visit Mode: **{predicted_mode}**")

        # Show recommendations for selected city
        st.write("### 🎯 Top Attractions in Selected City")
        recommended = df[
            (df['Continent'] == continent) &
            (df['Country'] == country) &
            (df['City'] == city)
        ][['Attraction', 'Rating']].drop_duplicates().sort_values(by='Rating', ascending=False)

        st.dataframe(recommended)

        # Show overall top attractions bar chart
        st.write("### 📊 Overall Top Attractions")
        top_attractions = df.groupby("Attraction")["Rating"].mean().sort_values(ascending=False).head(10)
        st.bar_chart(top_attractions)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
