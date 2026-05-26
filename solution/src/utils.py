"""
Generic utility functions
"""
import ast
import pandas as pd


def str_to_list(s):
    """
    Function to transform a string to a list
    """
    if isinstance(s, str) and s.startswith("["):
        try:
            return ast.literal_eval(s)
        except(ValueError, SyntaxError):
            return []
    return s if isinstance(s, list) else []


def load_data(tables, data_path="data/"):
    files = {
        "details": ("parquet", "details_clean.parquet"),
        "characters": ("parquet", "characters_clean.parquet"),
        "character_anime_works": ("parquet", "character_anime_works_clean.parquet"),
        "favs": ("parquet", "favs_clean.parquet"),
        "ratings": ("parquet", "ratings_clean.parquet"),
        "recommendations": ("parquet", "recommendations_clean.parquet"),
        "stats": ("parquet", "stats_clean.parquet"),
        "profiles": ("csv", "profiles.csv"),
    }
    kwargs = {
        "profiles": {"dtype": {"username": "string", "gender": "string"}},
    }
    result = {}
    for name in tables:
        info = files.get(name)
        if info is None:
            print(f"[load_data] warning: '{name}' is not a known table, skipping")
            continue
        fmt, fname = info
        path = data_path + fname
        kw = kwargs.get(name, {})
        if fmt == "parquet":
            result[name] = pd.read_parquet(path, **kw)
        else:
            result[name] = pd.read_csv(path, **kw)
        print(f"{name}: {len(result[name]):,} righe")
    return result

