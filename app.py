import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="Starbucks Nutrition Predictor", page_icon="☕", layout="centered")
st.title("☕ Starbucks Drink Nutrition Predictor")
st.write("Apni coffee customize karein aur dekhein us mein kitni Calories, Protein aur Fat hain!")

# Load trained model and metadata
@st.cache_resource
def load_model():
    return joblib.load('starbucks_nutrition_model.pkl')

artifact = load_model()
model = artifact['model']
feature_cols = artifact['feature_cols']
target_cols = artifact['target_cols']
dropdown_options = artifact['dropdown_options']

st.sidebar.header("Customize Your Drink")

# User inputs dynamically generate honge jo dataset mein thay
user_inputs = {}
for col in feature_cols:
    user_inputs[col] = st.sidebar.selectbox(f"Select {col}", dropdown_options[col])

# Extra customizations (Sugar / Espresso shots adjustment)
st.sidebar.subheader("Extra Add-ons")
extra_sugar_spoons = st.sidebar.slider("Extra Sugar (Teaspoons)", 0, 10, 0)
extra_espresso = st.sidebar.slider("Extra Espresso Shots", 0, 5, 0)

# Predict button
if st.button("Calculate Nutrition 📊", use_container_width=True):
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    
    results = dict(zip(target_cols, prediction))
    
    # Extra sugar (approx 16 cal, 4g sugar per spoon) & Espresso (approx 5 cal, 75mg caffeine per shot)
    for key in results:
        if 'calorie' in key.lower():
            results[key] += (extra_sugar_spoons * 16) + (extra_espresso * 5)
        elif 'sugar' in key.lower() or 'carb' in key.lower():
            results[key] += (extra_sugar_spoons * 4)
        elif 'caffeine' in key.lower():
            results[key] += (extra_espresso * 75)

    st.subheader("🥤 Estimated Nutrition Facts")
    
    # Results ko clean metrics cards mein dikhana
    cols = st.columns(3)
    for idx, (nutrient, val) in enumerate(results.items()):
        with cols[idx % 3]:
            st.metric(label=nutrient, value=f"{max(0, val):.1f}")
