import streamlit as st
import numpy as np

st.set_page_config(page_title="Deep Learning Demo", layout="wide")

st.title("Deep Learning Models Demonstration")

menu = st.sidebar.selectbox(
    "Choose Model",
    [
        "Perceptron",
        "AND Gate",
        "XOR Gate"
    ]
)

def step(x):
    return 1 if x >= 0 else 0

# ---------------- PERCEPTRON ----------------
if menu == "Perceptron":

    st.header("Single Perceptron")

    x = st.number_input("Study Hours", value=3.0)

    w = 1
    b = -3

    z = w * x + b
    prediction = step(z)

    st.write("Prediction:", prediction)

    if prediction == 1:
        st.success("Positive Class")
    else:
        st.error("Negative Class")


# ---------------- AND GATE ----------------
elif menu == "AND Gate":

    st.header("AND Gate Simulation")

    x1 = st.selectbox("Input 1", [0, 1])
    x2 = st.selectbox("Input 2", [0, 1])

    weights = np.array([1, 1])
    bias = -1.5

    result = np.dot(weights, [x1, x2]) + bias
    prediction = step(result)

    st.subheader(f"Output = {prediction}")


# ---------------- XOR GATE ----------------
else:

    st.header("XOR Gate Using ANN")

    x1 = st.selectbox("First Input", [0, 1])
    x2 = st.selectbox("Second Input", [0, 1])

    w1, w2, b1 = 1, 1, -0.5
    w3, w4, b2 = 1, 1, -1.5
    w5, w6, b3 = 1, -2, -0.5

    h1 = step(w1*x1 + w2*x2 + b1)
    h2 = step(w3*x1 + w4*x2 + b2)

    y = step(w5*h1 + w6*h2 + b3)

    st.subheader(f"Output = {y}")
