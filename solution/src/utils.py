"""
Generic utility functions
"""
import ast
import pandas as pd
import matplotlib.pyplot as plt


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

def plot_comparison_side_by_side(top_viewed, top_rated, figsize=(14, 6)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left Chart: Views ---
    top_v = top_viewed.sort_values('total', ascending=True)
    ax1.barh(top_v['title'], top_v['total'] / 1_000_000, color='royalblue')
    
    for i, (_, row) in enumerate(top_v.iterrows()):
        ax1.text((row['total'] / 1_000_000) * 1.02, i, f'Score: {row["score"]:.2f}', va='center')
        
    ax1.set_title('Top 10 by Views')
    ax1.set_xlabel('Total Members (Millions)')

    # --- Right Chart: Scores ---
    top_r = top_rated.sort_values('score', ascending=True)
    ax2.barh(top_r['title'], top_r['score'], color='crimson')
    
    for i, (_, row) in enumerate(top_r.iterrows()):
        ax2.text(row['score'] * 1.01, i, f'Votes: {row["scored_by"] / 1000:.0f}k', va='center')
        
    ax2.set_title('Top 10 by Score')
    ax2.set_xlabel('Score')

    fig.suptitle('Comparison: Most Watched vs Most Appreciated Anime', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()