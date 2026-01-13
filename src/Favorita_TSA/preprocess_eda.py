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


def save_table(df: pd.DataFrame, name: str):
    parquet_save(df, name)


def load_fact_table():
    return pd.read_parquet("data/processed/fact_table.parquet")


def load_table(name: str) -> pd.DataFrame:
    return pd.read_parquet(f"data/processed/{name}.parquet")


def aggregate(
    df: pd.DataFrame, group_cols: list[str], metrics: list[str]
) -> pd.DataFrame:
    return df.groupby(group_cols).agg(metrics).reset_index()


ALL_METRICS_UNIT_SALES = {
    "unit_sales": ["sum", "mean", "std", "max"],
}

# store_daily = aggregate(df_fact, ["store_nbr", "date"], metrics)
# item_daily = aggregate(df_fact, ["item_nbr", "date"], metrics)
# store_item_daily = aggregate(df_fact, ["store_nbr", "item_nbr", "date"], metrics)


def agg_daily(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["date"], ALL_METRICS_UNIT_SALES)


def agg_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["week"], ALL_METRICS_UNIT_SALES)


def agg_monthly(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["month"], ALL_METRICS_UNIT_SALES)


def store_daily(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["store_nbr", "date"], ALL_METRICS_UNIT_SALES)


def item_daily(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["item_nbr", "date"], ALL_METRICS_UNIT_SALES)


def store_item_daily(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["store_nbr", "item_nbr", "date"], ALL_METRICS_UNIT_SALES)


def store_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["store_nbr", "week"], ALL_METRICS_UNIT_SALES)


def item_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["item_nbr", "week"], ALL_METRICS_UNIT_SALES)


def store_item_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["store_nbr", "item_nbr", "week"], ALL_METRICS_UNIT_SALES)


def store_monthly(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["store_nbr", "month"], ALL_METRICS_UNIT_SALES)


def item_monthly(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["item_nbr", "month"], ALL_METRICS_UNIT_SALES)


def store_item_monthly(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate(df, ["store_nbr", "item_nbr", "month"], ALL_METRICS_UNIT_SALES)


df_fact = load_fact_table()

""" print("ITEM_DAILY \n", item_daily(df_fact))
print("STORE_DAILY \n", store_daily(df_fact))
print("STORE_ITEM_DAILY \n", store_item_daily(df_fact))
print("ITEM_WEEKLY \n", item_weekly(df_fact))
print("STORE_WEEKLY \n", store_weekly(df_fact))
print("STORE_ITEM_WEEKLY \n", store_item_weekly(df_fact))
print("ITEM_MONTHLY \n", item_monthly(df_fact))
print("STORE_MONTHLY \n", store_monthly(df_fact))
print("STORE_ITEM_MONTHLY \n", store_item_monthly(df_fact)) """


def save_dailys():
    save_table(item_daily(df_fact), "item_daily")
    save_table(store_daily(df_fact), "store_daily")
    save_table(store_item_daily(df_fact), "store_item_daily")


def save_weeklys():
    save_table(item_weekly(df_fact), "item_weekly")
    save_table(store_weekly(df_fact), "store_weekly")
    save_table(store_item_weekly(df_fact), "store_item_weekly")


def save_monthlys():
    save_table(item_monthly(df_fact), "item_monthly")
    save_table(store_monthly(df_fact), "store_monthly")
    save_table(store_item_monthly(df_fact), "store_item_monthly")


save_weeklys()
save_monthlys()


# Item Level - Daily, Weekly, Monthly Aggregations
# Store Level - Daily, Weekly, Monthly Aggregations
# Item + Store Level - Daily, Weekly, Monthly Aggregations
