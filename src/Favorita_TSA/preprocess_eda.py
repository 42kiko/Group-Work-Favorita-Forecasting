import pandas as pd

from Favorita_TSA.data_loader import parquet_loader, parquet_save
from Favorita_TSA.dataset import Dataset


def create_fact_table():
    df_train = parquet_loader(Dataset.TRAIN)
    df_fact = df_train.copy()
    df_fact["date"] = pd.to_datetime(df_fact["date"])
    df_fact["year"] = df_fact["date"].dt.year
    df_fact["month"] = df_fact["date"].dt.to_period("M")
    df_fact["week"] = df_fact["date"].dt.to_period("W")
    df_fact["dow"] = df_fact["date"].dt.dayofweek
    return df_fact


def save_fact_table():
    parquet_save(create_fact_table(), "fact_table")


def load_fact_table():
    return pd.read_parquet("data/processed/fact_table.parquet")


def aggregate(
    df: pd.DataFrame, group_cols: list[str], metrics: list[str]
) -> pd.DataFrame:
    return df.groupby(group_cols).agg(metrics).reset_index()


# Item Level - Daily, Weekly, Monthly Aggregations
# Store Level - Daily, Weekly, Monthly Aggregations
# Item + Store Level - Daily, Weekly, Monthly Aggregations
