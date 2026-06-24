import os

import pandas as pd

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


def is_allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def read_dataset(path: str) -> pd.DataFrame:
    """
    Читает CSV/XLSX без сокращения строк.
    Если CSV сохранён в другой кодировке, пробует несколько популярных вариантов.
    """
    _, ext = os.path.splitext(path.lower())

    if ext == ".csv":
        encodings = ["utf-8", "utf-8-sig", "cp1251", "latin1"]

        last_error = None

        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                return prepare_dataframe(df)
            except UnicodeDecodeError as e:
                last_error = e

        raise ValueError(f"Не удалось прочитать CSV из-за кодировки: {last_error}")

    if ext == ".xlsx":
        df = pd.read_excel(path)
        return prepare_dataframe(df)

    raise ValueError("Поддерживаются только CSV и XLSX файлы.")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Лёгкая подготовка таблицы:
    - убирает полностью пустые строки/колонки;
    - чистит пробелы в названиях колонок;
    - не сокращает датасет.
    """
    df = df.copy()

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    df.columns = [str(col).strip() for col in df.columns]

    return df


def get_dataset_preview(df: pd.DataFrame, rows: int = 10) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "preview_html": df.head(rows).to_html(classes="table", index=False),
    }