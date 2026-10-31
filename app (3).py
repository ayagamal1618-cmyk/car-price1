import numpy as np
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")


@st.cache_resource
def load_model():
    d = np.load("model_data.npz")
    return {k: d[k] for k in d.files}


def predict_log_price(m, x):
    """Walk the decision tree with plain numpy (same result as scikit-learn)."""
    x = x.astype(np.float32)
    node = 0
    while m["left"][node] != -1:
        if x[m["feature"][node]] <= m["threshold"][node]:
            node = m["left"][node]
        else:
            node = m["right"][node]
    return m["value"][node]


m = load_model()
cols = [str(c) for c in m["columns"]]


def opts(prefix):
    return [c[len(prefix):] for c in cols if c.startswith(prefix)]


st.title("🚗 Car Price Predictor")

make = st.selectbox("Make", opts("make_"))
year = st.number_input("Year", 1990, 2026, 2016)
hp = st.number_input("Engine HP", 50, 1200, 200)
size = st.selectbox("Vehicle size", opts("vehicle size_"))
trans = st.selectbox("Transmission", opts("transmission type_"))

if st.button("Predict price", use_container_width=True):
    x = np.zeros(len(cols))
    x[cols.index("year")] = year
    x[cols.index("engine hp")] = hp
    x[cols.index(f"make_{make}")] = 1
    x[cols.index(f"vehicle size_{size}")] = 1
    x[cols.index(f"transmission type_{trans}")] = 1
    price = float(np.expm1(predict_log_price(m, x)))  # model predicts log(price)
    st.metric("Estimated price", f"${price:,.0f}")
