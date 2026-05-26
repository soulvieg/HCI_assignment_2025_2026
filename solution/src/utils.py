"""
Generic utility functions
"""
import ast
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from ipywidgets import interact, Dropdown


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


def plot_hbar(data, label_col, val_col, color_col=None, annot_col=None,
              annot_fmt="{:.2f}", annot_prefix="", annot_suffix="",
              title="", xlabel="", figsize=(10, 6),
              cmap="viridis", top_n=None, sort_by=None):
    if sort_by is None:
        sort_by = val_col
    if top_n is not None:
        data = data.nlargest(top_n, sort_by)

    data = data.sort_values(sort_by, ascending=True).copy()

    fig, ax = plt.subplots(figsize=figsize)

    if color_col is not None:
        norm = plt.Normalize(data[color_col].min(), data[color_col].max())
        colors = plt.colormaps[cmap](norm(data[color_col]))
    else:
        colors = None

    bars = ax.barh(data[label_col], data[val_col], color=colors)

    if annot_col is not None:
        for i, (_, row) in enumerate(data.iterrows()):
            label = annot_prefix + annot_fmt.format(row[annot_col]) + annot_suffix
            ax.text(row[val_col] * 1.01, i, label, va="center")

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def show_genre_explorer(df):
    """
    Generates an interactive Plotly bar chart for top anime by genre.
    """
    genres_list = sorted(df['genres'].dropna().unique())
    genre_dropdown = Dropdown(options=genres_list, description='Genre:', layout={'width': '300px'})

    def plot_genre_top(genre):
        # Use the passed 'df' instead of a global variable
        data = df[df['genres'] == genre].nlargest(10, 'score')
        
        fig = px.bar(data, x='score', y='title', orientation='h',
                     title=f'Top 10 Anime — {genre}',
                     color='score', color_continuous_scale='viridis',
                     text='score')
        
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        
        # Calculate dynamic range
        min_score = data['score'].min() - 0.05
        max_score = data['score'].max() + 0.05
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'}, 
            xaxis={'range': [min_score, max_score]}, 
            height=500
        )
        fig.show()

    interact(plot_genre_top, genre=genre_dropdown)