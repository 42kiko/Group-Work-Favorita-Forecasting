import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

llm = LocalLLM(api_base="http://localhost:11434/v1", model="llama3")

st.title("Data analysis with PandasAI")

uploaded_file = st.file_uploader(
    "Upload a CSV or Parquet file", type=["parquet", "csv"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".parquet"):
        data = pd.read_parquet(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    df = SmartDataframe(data, config={"llm": llm})

    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate") and prompt:
        with st.spinner("Generating response..."):
            st.write(df.chat(prompt))
