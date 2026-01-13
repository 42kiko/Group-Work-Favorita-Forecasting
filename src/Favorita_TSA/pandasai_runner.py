import pandasai as pai
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM

from Favorita_TSA.preprocess_eda import load_table

llm = LiteLLM(
    model="ollama/mistral",  # oder "ollama/mistral"
    api_base="http://localhost:11434",
    api_key="EMPTY",
)

pai.config.set({"llm": llm})
store_daily = load_table("store_daily")

sdf = SmartDataframe(store_daily)
print(sdf.chat("What is the average unit_sales for store_nbr 1 in 2017?"))
