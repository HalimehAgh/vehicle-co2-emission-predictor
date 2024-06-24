import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load data
file_path_df = 'Clean_Data_Preprocessed.csv'
df = pd.read_csv(file_path_df)

# Sidebar
st.sidebar.header("CO2 Car Emission Prediction")
st.sidebar.image('DataScientest_logo.jpg')
st.sidebar.write("This dashboard uses CO emission data from 2013 to predict car emissions.")
st.sidebar.write("")
st.sidebar.markdown("**Authors:**\n- Halimeh\n- Pavel\n- Abd Akdim")

# Main body
st.title("CO2 Car Emission Prediction Dashboard")

# Display dataset
st.header("Dataset")
st.write("Here is a preview of the dataset used in this analysis:")
st.dataframe(df.head())

# Descriptive statistics
st.header("Descriptive Statistics")
st.write("Summary statistics of the dataset:")
st.write(df.describe())

# Data visualizations
st.header("Data Visualizations")

# Histogram of CO2 emissions
st.subheader("CO2 Emissions Distribution")
fig, ax = plt.subplots()
sns.histplot(df['CO2 (g/km)'], kde=True, ax=ax)  #
ax.set_title('Distribution of CO2 Emissions')
ax.set_xlabel('CO2 Emissions (g/km)')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Scatter Plot: CO2 Emissions vs. User-selected Variable
st.subheader("CO2 Emissions vs. User-selected Variable")

# Dropdown menu for user to select variable
x_axis = st.selectbox("Select variable for X-axis", df.columns)

fig = px.scatter(df, x=x_axis, y='CO2 (g/km)', title=f'CO2 Emissions vs. {x_axis}')
st.plotly_chart(fig)


# # Display correlation matrix
# st.subheader("Correlation Matrix")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
# ax.set_title('Correlation Matrix')
# st.pyplot(fig)


# Comparison of Different Model Metrics

st.subheader("Comparison of Different Model Metrics")
# Select Models and Metrics
# Load data from CSV
file_path_df = 'df_metrics.csv'
df_metrics = pd.read_csv(file_path_df)
selected_models = st.multiselect('Select Models', df_metrics['Model'].unique())
selected_metrics = st.multiselect('Select Metrics', df_metrics.columns[1:])  # Exclude 'Model'

# Filter Data based on Selection
filtered_df = df_metrics[df_metrics['Model'].isin(selected_models)]

# Create Bar Chart with Plotly
if not filtered_df.empty:
    fig = px.bar(filtered_df, x='Model', y=selected_metrics,
                 labels={'Model': 'Models', 'value': 'Metric Value', 'variable': 'Metric'},
                 title='Comparison of Selected Model Metrics',
                 barmode='group')
    
    # Layout Adjustments
    fig.update_layout(xaxis={'categoryorder': 'total descending'})  # Sort x-axis by total descending
    
    # Display Chart in Streamlit
    st.plotly_chart(fig)
else:
    st.error('No data available for the selected models.')






