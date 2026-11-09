import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.set_page_config(page_title="Palestine Casualty Prediction & Analysis", page_icon="📊", layout="wide")

st.title("📊 Palestine Body Count & Casualty Prediction System")
st.markdown("An interactive machine learning and data analysis platform tracking historical conflicts and predicting casualty trends.")

@st.cache_resource
def load_model():
    with open('best_palestine_model.pkl', 'rb') as f:
        model, imputer = pickle.load(f)
    return model, imputer

try:
    model, imputer = load_model()
except Exception as e:
    st.error(f"⚠️ Model file not found! Please train and save 'best_palestine_model.pkl' first. Error: {e}")

@st.cache_data
def load_data():
    df = pd.read_csv('Palestine Body Count.csv')
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
    
    cols_to_clean = ['Palestinians Injuries', 'Israelis Injuries', 'Palestinians Killed', 'Israelis Killed']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)


    st.sidebar.header("🧭 Navigation & Inputs")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Interactive Dashboard (EDA)", "Casualty Prediction Model"])

    if app_mode == "Interactive Dashboard (EDA)":
        st.subheader("📈 Historical Trends & Conflict Analysis")
        
        
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))
        filtered_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🇵🇸 Palestinians Killed", f"{int(filtered_df['Palestinians Killed'].sum()):,}")
        col2.metric("🇮🇱 Israelis Killed", f"{int(filtered_df['Israelis Killed'].sum()):,}")
        col3.metric("🇵🇸 Palestinians Injured", f"{int(filtered_df['Palestinians Injuries'].sum()):,}")
        col4.metric("🇮🇱 Israelis Injured", f"{int(filtered_df['Israelis Injuries'].sum()):,}")

        st.markdown("---")

        
        c1, c2 = st.columns(2)
        with c1:
            timeline_df = filtered_df.groupby('Year')[['Palestinians Killed', 'Israelis Killed']].sum().reset_index()
            fig_time = px.line(timeline_df, x='Year', y=['Palestinians Killed', 'Israelis Killed'], title="Fatalities Over Time", template="plotly_dark")
            st.plotly_chart(fig_time, use_container_width=True)
            
        with c2:
            injury_df = filtered_df.groupby('Year')[['Palestinians Injuries', 'Israelis Injuries']].sum().reset_index()
            fig_inj = px.bar(injury_df, x='Year', y=['Palestinians Injuries', 'Israelis Injuries'], title="Injuries Over Time", template="plotly_dark")
            st.plotly_chart(fig_inj, use_container_width=True)

    else:
        st.subheader("🤖 Real-Time Casualty Prediction System")
        st.write("Enter the parameters below to predict conflict impact using our trained machine learning model.")

        col1, col2 = st.columns(2)
        with col1:
            input_year = st.number_input("Year", min_value=2000, max_value=2030, value=2026)
            input_month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
            rolling_killed = st.number_input("Average Killed in Past 3 Months", min_value=0.0, value=50.0)
        with col2:
            rolling_injured = st.number_input("Average Injured in Past 3 Months", min_value=0.0, value=150.0)
            lag_1 = st.number_input("Previous Month Killed", min_value=0.0, value=45.0)
            lag_2 = st.number_input("2 Months Ago Killed", min_value=0.0, value=40.0)

        if st.button("🔮 Predict Casualties", use_container_width=True):
            user_data = np.array([[input_year, input_month, rolling_killed, rolling_injured, lag_1, lag_2]])
            user_data_imputed = imputer.transform(user_data)
            prediction = model.predict(user_data_imputed)
            
            st.success(f"### 🎯 Estimated Palestinians Killed: **{int(prediction[0])}**")

except Exception as e:
    st.error(f"⚠️ Error running dashboard: {e}")