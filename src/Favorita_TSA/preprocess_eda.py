import pandas as pd

from Favorita_TSA.utils.data_loader import df_to_parquet, parquet_loader, parquet_save
from Favorita_TSA.utils.dataset import Dataset, PreDataset


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


def fix_agg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens MultiIndex / tuple columns into snake_case strings.

    Examples:
    ('unit_sales', 'sum') -> unit_sales_sum
    ('store_nbr', '')     -> store_nbr
    'date'                -> date
    """
    df = df.copy()

    def clean(col: object) -> str:
        # Case 1: tuple column from groupby/agg
        if isinstance(col, tuple):
            parts = [str(p) for p in col if p]
            return "_".join(parts)

        # Case 2: already string
        if isinstance(col, str):
            return col

        # Fallback (should not happen)
        return str(col)

    df.columns = [clean(c) for c in df.columns]
    return df


def save_table(df: pd.DataFrame, name: PreDataset) -> None:
    parquet_save_prepocessed(fix_agg_columns(df), name)


def parquet_save_prepocessed(df: pd.DataFrame, name: PreDataset) -> None:
    df_to_parquet(
        df,
        f"data/processed/preprocessed/{name.value}.parquet",
    )


def load_fact_table():
    return pd.read_parquet("data/processed/preprocessed/fact_table.parquet")


def load_table(name: PreDataset) -> pd.DataFrame:
    return pd.read_parquet(f"data/processed/preprocessed/{name.value}.parquet")


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


def save_dailys():
    df_fact = load_fact_table()
    save_table(item_daily(df_fact), PreDataset.ITEM_DAILY)
    save_table(store_daily(df_fact), PreDataset.STORE_DAILY)
    save_table(store_item_daily(df_fact), PreDataset.STORE_ITEM_DAILY)


def save_weeklys():
    df_fact = load_fact_table()
    save_table(item_weekly(df_fact), PreDataset.ITEM_WEEKLY)
    save_table(store_weekly(df_fact), PreDataset.STORE_WEEKLY)
    save_table(store_item_weekly(df_fact), PreDataset.STORE_ITEM_WEEKLY)


def save_monthlys():
    df_fact = load_fact_table()
    save_table(item_monthly(df_fact), PreDataset.ITEM_MONTHLY)
    save_table(store_monthly(df_fact), PreDataset.STORE_MONTHLY)
    save_table(store_item_monthly(df_fact), PreDataset.STORE_ITEM_MONTHLY)


# save_dailys()
# save_weeklys()
# save_monthlys()


# Item Level - Daily, Weekly, Monthly Aggregations
# Store Level - Daily, Weekly, Monthly Aggregations
# Item + Store Level - Daily, Weekly, Monthly Aggregations
