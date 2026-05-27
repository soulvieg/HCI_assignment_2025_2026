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
        "person_voice_works": ("parquet", "person_voice_works_clean.parquet"),
        "person_details": ("parquet", "person_details_clean.parquet"),
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


def _genre_agg(details_genres):
    return details_genres.groupby('genres').agg(
        anime_count=('mal_id', 'nunique'),
        avg_score=('score', 'mean'),
        total_members=('members', 'sum'),
        avg_members=('members', 'mean'),
        median_score=('score', 'median'),
        std_score=('score', 'std')
    ).reset_index()

def plot_genre_scatter_volume_quality(details_genres):
    genre_agg = _genre_agg(details_genres)
    global_avg_score = details_genres['score'].mean()
    global_avg_count = genre_agg['anime_count'].mean()
    fig = px.scatter(
        genre_agg,
        x='anime_count',
        y='avg_score',
        size='total_members',
        text='genres',
        hover_data={'anime_count': True, 'avg_score': ':.2f',
                    'total_members': True, 'median_score': ':.2f',
                    'std_score': ':.2f', 'avg_members': True},
        title='Volume vs. Quality: Which Genres Perform Best?',
        labels={'anime_count': '# Anime Produced',
                'avg_score': 'Average Score',
                'total_members': 'Total Members'},
        template='plotly_white',
        color='avg_score',
        color_continuous_scale='Viridis',
        size_max=60
    )
    fig.add_hline(
        y=global_avg_score, line_dash='dash', line_color='red', opacity=0.5,
        annotation_text=f'Global avg score: {global_avg_score:.2f}',
        annotation_position='bottom right'
    )
    fig.add_vline(
        x=global_avg_count, line_dash='dash', line_color='gray', opacity=0.5,
        annotation_text=f'Avg production: {global_avg_count:.0f}',
        annotation_position='top left'
    )
    fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(height=650, font=dict(size=12))
    fig.show()


def plot_studio_consistency(details_df: pd.DataFrame, min_anime_count: int = 1):
    """
    Generates an interactive Plotly scatter plot analyzing studio quality vs. consistency.
    """

    details_studios = details_df.explode("studios").dropna(subset=["studios", "score"])

    studio_stats = details_studios.groupby("studios").agg(
        num_anime=("mal_id", "count"),
        avg_score=("score", "mean"),
        score_std=("score", "std")
    ).reset_index()

    titans = studio_stats[studio_stats["num_anime"] >= min_anime_count].copy()
    
    titans["score_std"] = titans["score_std"].fillna(0)

    fig = px.scatter(
        titans,
        x="avg_score",
        y="score_std",
        size="num_anime",
        hover_name="studios", 
        hover_data={
            "num_anime": True, 
            "avg_score": ':.2f', 
            "score_std": ':.2f'
        },
        title="The Monopoly of the Titans: Studio Consistency vs. Quality",
        labels={
            "avg_score": "Average Score (Higher = Better Quality)",
            "score_std": "Score Variance / Std Dev (Lower = More Consistent)"
        },
        size_max=55,
        opacity=0.6,
        color_discrete_sequence=["#3498db"] 
    )

    fig.add_hline(y=titans['score_std'].median(), line_dash="dot", line_color="gray", annotation_text="Median Consistency")
    fig.add_vline(x=titans['avg_score'].median(), line_dash="dot", line_color="gray", annotation_text="Median Score")

    # Display the plot
    fig.show()

def plot_violin_star_power(person_voice_works, person_details, details):
    voice_actors = person_voice_works.merge(
        person_details[['person_mal_id', 'favorites']],
        on='person_mal_id', how='inner'
    )
    anime_star_power = voice_actors.groupby('anime_mal_id').agg(
        avg_star_power=('favorites', 'mean'),
        cast_size=('person_mal_id', 'nunique')
    ).reset_index()
    star_system = anime_star_power.merge(
        details[['mal_id', 'title', 'score', 'scored_by']],
        left_on='anime_mal_id', right_on='mal_id', how='inner'
    )
    df_clean = star_system[
        star_system['score'].notna() & (star_system['scored_by'] >= 1000)
    ].copy()
    labels = ['1. Niche / Unknown', '2. Standard Cast',
              '3. Popular Cast', '4. All-Star Cast']
    df_clean['cast_tier'] = pd.qcut(df_clean['avg_star_power'], q=4, labels=labels)
    fig = px.violin(
        df_clean, x="cast_tier", y="score", color="cast_tier",
        box=True, points="outliers",
        hover_data=["title", "avg_star_power", "cast_size"],
        title="Voices of Gold: Does Star Power Guarantee a Hit?",
        labels={
            "cast_tier": "Cast Star Power Tier (Based on Avg Seiyuu Favorites)",
            "score": "Anime Score"
        },
        template="plotly_white",
        category_orders={"cast_tier": labels}
    )
    fig.update_layout(showlegend=False)
    fig.show()


def plot_character_popularity_scatter(characters, character_anime_works, details):
    """
    Creates a scatter plot comparing Anime Popularity to Character Favorites.
    """
    char_roles = character_anime_works.merge(
        characters[['character_mal_id', 'name', 'favorites']], 
        left_on='character_mal_id', 
        right_on='character_mal_id', 
        how='inner'
    )
    
    # Merge with anime details to get anime popularity (using members or scored_by)
    full_df = char_roles.merge(
        details[['mal_id', 'title', 'members']], 
        left_on='anime_mal_id', 
        right_on='mal_id', 
        how='inner',
        suffixes=('_char', '_anime')
    )

    full_df_sorted = full_df.sort_values(by='members', ascending=False)

    unique_chars = full_df_sorted.drop_duplicates(subset=['character_mal_id'], keep='first')

    # To avoid massive clutter, let's look at the Top 500 or 1000 most favorited characters
    top_chars = unique_chars.sort_values(by='favorites', ascending=False).head(1000)
    

    fig = px.scatter(
        top_chars, 
        x="members", 
        y="favorites", 
        color="role",
        hover_name="name",
        hover_data={"title": True, "members": True, "favorites": True, "role": False},
        labels={
            "members": "Anime Popularity (Total Members)",
            "favorites": "Character Favorites",
            "role": "Character Role"
        },
        title="Do Popular Anime Always Produce the Most Loved Characters?",
        opacity=0.7,
        color_discrete_map={"Main": "#EF553B", "Supporting": "#636EFA"} # Distinct colors
    )
    
    # Use log scale because popularity and favorites usually have massive outliers
    fig.update_layout(
        xaxis_type="log", 
        yaxis_type="log",
        template="plotly_white",
        legend_title="Role"
    )
    
    fig.show()

def plot_role_favorites_violin(characters, character_anime_works):
    """
    Creates a violin plot comparing the distribution of favorites between Main and Supporting characters.
    """

    df = character_anime_works.merge(
        characters[['character_mal_id', 'name', 'favorites']], 
        left_on='character_mal_id', 
        right_on='character_mal_id', 
        how='inner'
    )
    
    # Filter out characters with 0 favorites to get a cleaner distribution of actually "liked" characters
    df_clean = df[df['favorites'] > 10].copy()
    
    # Filter to only Main and Supporting
    df_clean = df_clean[df_clean['role'].isin(['Main', 'Supporting'])]
    
    fig = px.violin(
        df_clean, 
        x="role", 
        y="favorites", 
        color="role",
        box=True, 
        points=False, 
        hover_name="name",
        labels={
            "role": "Character Role",
            "favorites": "Total Favorites (Log Scale)"
        },
        title="Main vs. Supporting: Who Really Captures the Audience's Heart?",
        color_discrete_map={"Main": "#EF553B", "Supporting": "#636EFA"}
    )
    
    # Log scale on Y-axis because a few characters will have 100k+ favorites, while most have < 100
    fig.update_layout(
        yaxis_type="log",
        template="plotly_white",
        showlegend=False # Legend is redundant since x-axis has the role names
    )
    
    fig.show()