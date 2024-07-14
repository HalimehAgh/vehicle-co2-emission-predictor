import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import os


logo_path = 'logo.png'  # logo path

cover_image_path = 'cover_image.png'  # Cover
svg_path = 'Methodology.svg'  # SVG Methodology

# Chapter titles
chapters = ["Introduction", "Data mining and Visualization", "Pre-processing","Feature Engineering","Modeling","Interpretation of results","Conclusion", "Application"]

# Sidebar with logo and table of contents
st.sidebar.image(logo_path, use_column_width=True)
st.sidebar.title("Agenda")
selected_chapter = st.sidebar.radio("", ["Home"] + chapters)

# Display the cover page by default
if selected_chapter == "Home":
    st.title("CO2 Emissions Prediction Project")
    st.image(cover_image_path, use_column_width=True)
    st.write("### Team Members")
    st.write("""
    - Abd Akdim
    - Halimeh Agh
    - Halimeh Agh
    """)
    st.write("### Supervisor")
    st.write("""
    - Sarah Lenet
    """)

# Introduction
if selected_chapter == "Introduction":
    st.title("Introduction")
    st.write("""
    ****Objectiv:****\n
    The goal of this project is to develop a model to predict the CO2 emissions of cars based on vehicle characteristics to assess the environmental impact.
 

    """)
    st.title("Methodology")
    st.write("""
    -
    -
    -
    - 
     """)
    # SVG
    st.image(svg_path, width=800) 
   

# Data mining and Visualization
elif selected_chapter == "Data mining and Visualization":
    st.title("Data mining and Visualization")
    st.write("""
    
    """)

# Pre-processing
elif selected_chapter == "Pre-processing":
    st.title("Pre-processing")
    st.write("""
    
    """)


# Feature Engineering
elif selected_chapter == "Feature Engineering":
    st.title("Feature Engineering")
    st.write("""
   
    """)

# Modeling
elif selected_chapter == "Modeling":
    st.title("Modeling")
    st.write("""
    .
    """)

# Interpretation of results
elif selected_chapter == "Interpretation of results":
    st.title("Interpretation of results")
    st.write("""

    """)

# Conclusion
elif selected_chapter == "Conclusion":
    st.title("Conclusion")
    st.write("""
    
    """)

# Application (Data Science Testing)
elif selected_chapter == "Application":
    st.title("Prediction of CO2 Emissions")
    st.write("""
    
    """)

    # Input form in the sidebar
    with st.sidebar:
        st.header("Input Data")
        Input_Data = {
            "Consommation extra-urbaine (l/100km)": st.number_input("Consommation extra-urbaine (l/100km)", min_value=0, max_value=100),
            "Consommation mixte (l/100km)": st.number_input("Consommation mixte (l/100km)", min_value=0, max_value=100),
            "NOX (g/km)": st.number_input("NOX (g/km)", min_value=0, max_value=100),
            "Consommation urbaine (l/100km)": st.number_input("Consommation urbaine (l/100km)", min_value=0, max_value=100),
            "Carburant_GO": 1 if st.toggle("Carburant_GO - Yes/No", key="carburant_go") else 0,
            "Carburant_ES": 1 if st.toggle("Carburant_ES - Yes/No", key="carburant_es") else 0,
            "Puissance maximale (kW)": st.number_input("Puissance maximale (kW)", min_value=0, max_value=500),
            "Puissance administrative": st.number_input("Puissance administrative", min_value=0, max_value=500),
            "Mode UTAC_CADDY": 1 if st.toggle("Mode UTAC_CADDY - Yes/No", key="mode_utac_caddy") else 0,
            "CO type I (g/km)": st.number_input("CO type I (g/km)", min_value=0, max_value=100),
        }#
        with st.sidebar:
            st.subheader("Your Input Data")
            st.write(pd.DataFrame.from_dict(Input_Data, orient='index', columns=['Value']).style.set_table_attributes('class="full-width-table"'))

    # Load the saved model
    model = joblib.load('rf_model.pkl')

    # Retrieve the feature names used during training
    model_features = model.feature_names_in_

    # Ensure the input data has the same features as the model expects
    input_data = pd.DataFrame([Input_Data], columns=model_features).reindex(columns=model_features, fill_value=0)


    # Prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        co2_value = prediction[0]
        
        # Display the prediction in a styled box
        st.markdown(
            f"""
            <div style="background-color: rgb(255, 86, 68); padding: 20px; border-radius: 5px;">
                <h2 style="color: white; text-align: center;">Predicted CO2 Emissions: {co2_value:.2f} g/km</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

