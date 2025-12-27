from pathlib import Path

import pandas as pd


def load_train_csv(path: str | Path) -> pd.DataFrame:
    """
    Lädt die Trainingsdaten mit optimierten Typen:
    'boolean' (mit großem B) erlaubt 0, 1 und NA Werte.
    """
    return pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={
            "id": "int64",
            "item_nbr": "int32",
            "store_nbr": "int16",
            "family": "string",
            "onpromotion": "boolean",
        },
    )


def load_df(path: str | Path) -> pd.DataFrame:
    """Standard-Loader für kleinere Dataframes."""
    return pd.read_csv(path)


def df_to_parquet(df: pd.DataFrame, parquet_path: str | Path) -> None:
    """Speichert einen Dataframe als Parquet-Datei."""
    df.to_parquet(parquet_path, index=False)


def df_to_parquet_packages(df: pd.DataFrame, parquet_path: str | Path) -> None:
    """Speichert einen Dataframe als Parquet-Datei."""

    df.to_parquet(parquet_path, index=False)


def split_parquet_to_packages(
    df: pd.DataFrame, name: str, target_root: str | Path
) -> None:
    """
    Splittet df in Parquet Parts.
    Ziel: möglichst nah an target_mb, garantiert nicht größer als hard_limit_mb.
    """
    target_dir = Path(target_root) / name
    target_dir.mkdir(parents=True, exist_ok=True)

    total_rows = len(df)
    if total_rows == 0:
        print(f"📦 Keine Daten für: {name}")
        return

    target_bytes = int(99.0 * 1024 * 1024)
    hard_limit_bytes = int(99.0 * 1024 * 1024)
    s_rows = min(200_000, total_rows)
    s_start = max(0, (total_rows // 2) - (s_rows // 2))
    tmp = target_dir / "_sample_tmp.parquet"
    df.iloc[s_start : s_start + s_rows].to_parquet(tmp, index=False)
    bytes_per_row = max(1.0, tmp.stat().st_size / s_rows)
    tmp.unlink()
    rows_est = max(1, int(int(99.0 * 1024 * 1024) / bytes_per_row))

    start = 0
    part = 0
    while start < total_rows:
        end_guess = min(total_rows, start + rows_est)
        out_path = target_dir / f"part_{part:01d}.parquet"
        end_final, written_bytes = _write_part_with_hard_limit(
            df=df,
            start=start,
            end=end_guess,
            out_path=out_path,
            hard_limit_bytes=hard_limit_bytes,
        )

        actual_mb = written_bytes / (1024**2)
        print(f"   ✅ {name} Part {part}: {actual_mb:.2f} MB")
        if written_bytes > 0:
            ratio = target_bytes / written_bytes
            rows_est = max(1, int((end_final - start) * ratio))
        start = end_final
        part += 1

    del df
    print(f"📦 Pakete erfolgreich erstellt in: {target_dir}")


def _write_part_with_hard_limit(
    df: pd.DataFrame,
    start: int,
    end: int,
    out_path: Path,
    *,
    hard_limit_bytes: int,
) -> tuple[int, int]:
    """
    Schreibt df[start:end] nach out_path.
    Wenn die Datei größer als hard_limit_bytes ist, verkleinert end iterativ und schreibt neu.
    """
    end = max(start + 1, end)

    while True:
        df.iloc[start:end].to_parquet(out_path, index=False)
        size = out_path.stat().st_size

        if size <= hard_limit_bytes:
            return end, size

        rows = end - start
        if rows <= 1:
            return end, size

        # verkleinern mit Sicherheitsfaktor
        shrink_ratio = (hard_limit_bytes / size) * 0.98
        new_rows = max(1, int(rows * shrink_ratio))
        end = start + new_rows


def save_tables_to_parquet() -> None:
    """
    Geht die Liste der Tabellen durch und speichert sie als Parquet-Pakete.
    Train landet in ../data/train/, andere in ../data/name_pkg/.
    """
    data = {
        "items",
        "oil",
        "holidays_events",
        "stores",
        "test",
        "transactions",
        "train",
    }

    for element in data:
        if element == "train":
            split_parquet_to_packages(
                load_train_csv(f"data/raw/{element}.csv"),
                "train",
                f"data/processed/{element}.parquet",
            )
        else:
            df_to_parquet(
                load_df(f"data/raw/{element}.csv"), f"data/processed/{element}.parquet"
            )


save_tables_to_parquet()


# df_to_parquet(load_train_csv("data/raw/train.csv"), "data/processed/train.parquet")


# df_to_parquet(load_df("data/raw/items.csv"), "data/processed/items.parquet")


# csv_to_parquet("data/raw/train.csv", "data/processed/train.parquet")
