import pandas as pd
from ydata_profiling import ProfileReport

from Favorita_TSA.data_loader import parquet_loader
from Favorita_TSA.dataset import Dataset

all_data: dict[Dataset, pd.DataFrame] = {
    Dataset.OIL: parquet_loader(Dataset.OIL.value),
    Dataset.ITEMS: parquet_loader(Dataset.ITEMS.value),
    Dataset.HOLIDAYS_EVENTS: parquet_loader(Dataset.HOLIDAYS_EVENTS.value),
    Dataset.STORES: parquet_loader(Dataset.STORES.value),
    Dataset.TRANSACTIONS: parquet_loader(Dataset.TRANSACTIONS.value),
    Dataset.TRAIN: parquet_loader(Dataset.TRAIN.value),
}


for dataset, dataframe in all_data.items():
    profile = ProfileReport(
        dataframe, title=f"YData Profiling Report - {dataset.value}", explorative=True
    )
    profile.to_file(f"./reports/{dataset.value}_report.html")


# profile = ProfileReport(df_items, title="YData Profiling Report", explorative=True)
