import streamlit as st
import pandas as pd
import plotly.express as px

st.title("AI Posture Monitoring Dashboard")

# Load posture data
df = pd.read_csv("posture_log.csv")

# Convert timestamp
df["time"] = pd.to_datetime(df["timestamp"], unit="s")

st.subheader("Raw Posture Data")
st.dataframe(df)

# Posture distribution
posture_counts = df["posture"].value_counts()

fig1 = px.pie(
    values=posture_counts.values,
    names=posture_counts.index,
    title="Posture Distribution"
)

st.plotly_chart(fig1)

# Spine angle over time
fig2 = px.line(
    df,
    x="time",
    y="angle",
    title="Spine Angle Over Time"
)

st.plotly_chart(fig2)

# Posture score
good = posture_counts.get("Good Posture", 0)
total = len(df)

score = (good / total) * 100 if total > 0 else 0

st.metric("Posture Score", f"{score:.2f}%")

slouch = posture_counts.get("Slouching", 0)
st.metric("Total Slouch Frames", slouch)

st.write("AI posture monitoring insights generated from real-time detection.")
