from enum import Enum


class Dataset(str, Enum):
    OIL = "oil"
    ITEMS = "items"
    HOLIDAYS_EVENTS = "holidays_events"
    STORES = "stores"
    TRANSACTIONS = "transactions"
    TRAIN = "train"
    TEST = "test"
