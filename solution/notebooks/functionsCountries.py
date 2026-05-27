import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import time
import gc

#---load
def loadUserRatings():

    profiles = pd.read_parquet(
        '../data/profiles_clean.parquet',
        engine='pyarrow',
        columns=['username', 'location']
    )

    location_map = profiles.set_index('username')['location'].astype('category')

    del profiles
    gc.collect()

    user_ratings = pd.read_parquet(
        '../data/ratings_clean.parquet',
        engine='pyarrow',
        columns=['username', 'anime_id', 'score']
    )

    user_ratings['location'] = user_ratings['username'].map(location_map)

    user_ratings.drop(columns=['username'], inplace=True)

    user_ratings.dropna(subset=['location'], inplace=True)

    return user_ratings

#---1
def topCountriesVotes(user_ratings, top_n=20):

    country_counts = user_ratings['location'].value_counts().reset_index()
    country_counts.columns = ['location', 'total_votes']

    country_counts = country_counts[~country_counts['location'].isin(['Unknown', ''])]

    top_20_countries = country_counts.head(20)

    fig = px.bar(
    top_20_countries,
    x='location',
    y='total_votes',
    title='Top 20 Countries by Total Anime Votes',
    labels={'location': 'Country', 'total_votes': 'Total Votes'},
    color='total_votes',
    color_continuous_scale='Sunsetdark',
    text_auto='.2s'
    )

    fig.update_layout(
    xaxis_tickangle=-45,
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(size=14)
    )

    fig.show()

#---2
def globalAnimeUserDistribution(user_ratings):

    location_totals = user_ratings['location'].value_counts().reset_index()

    location_totals.columns = ['location', 'total_votes']

    valid_mask = ~location_totals['location'].isin(['Unknown', ''])

    map_data = location_totals[valid_mask].copy()

    map_data['log_votes'] = np.log10(map_data['total_votes'])

    fig = px.choropleth(
        map_data,
        locations="location",
        locationmode="country names",
        color="log_votes",
        hover_name="location",
        hover_data={
            "total_votes": ':,.0f',
            "log_votes": False,
            "location": False
        },
        color_continuous_scale="Viridis",
        title="Global Anime User Base Distribution"
    )

    fig.update_layout(
        template="plotly_white",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        coloraxis_colorbar=dict(
            title="Total Votes",
            tickvals=[2, 3, 4, 5, 6, 7],
            ticktext=['100', '1k', '10k', '100k', '1M', '10M']
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#333333"
        ),
        title_font=dict(
            size=18,
            family="Arial, sans-serif"
        ),
        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0
        )
    )

    fig.show()

#---3
def globalAnimeAverageScore(user_ratings, significance_threshold=10000):

    country_stats = user_ratings.groupby('location').agg(
        average_score=('score', 'mean'),
        total_votes=('score', 'count')
    ).reset_index()

    valid_mask = (
        (~country_stats['location'].isin(['Unknown', ''])) &
        (country_stats['total_votes'] >= significance_threshold)
    )

    map_data = country_stats[valid_mask].copy()

    min_bound = map_data['average_score'].quantile(0.05)
    max_bound = map_data['average_score'].quantile(0.95)

    fig = px.choropleth(
        map_data,
        locations="location",
        locationmode="country names",
        color="average_score",
        range_color=[min_bound, max_bound],
        hover_name="location",
        hover_data={
            "average_score": ':.2f',
            "total_votes": ':,.0f',
            "location": False
        },
        color_continuous_scale="Plasma",
        title="Global Anime Rating Severity: Average Score per Country"
    )

    fig.update_layout(
        template="plotly_white",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        coloraxis_colorbar=dict(
            title="Avg Score",
            tickformat=".2f"
        ),
        font=dict(family="Arial, sans-serif", size=12, color="#333333"),
        title_font=dict(size=18, family="Arial, sans-serif"),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()

#---4
def violinTopLocations(user_ratings, sample_limit=1000000):

    top_5_locations = user_ratings['location'].value_counts().nlargest(5).index

    top_locations_data = user_ratings[
        user_ratings['location'].isin(top_5_locations)
    ]

    if len(top_locations_data) > sample_limit:
        plot_data = top_locations_data.sample(n=sample_limit, random_state=42)
    else:
        plot_data = top_locations_data

    plt.figure(figsize=(14, 8))

    sns.violinplot(
        data=plot_data,
        x='location',
        y='score',
        palette='coolwarm',
        order=top_5_locations,
        inner="quartile",
        cut=0,
        linewidth=1.5,
        bw_adjust=2.5
    )

    plt.title(
        'Density and Distribution of Votes in the 5 Most Active Countries',
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    plt.xlabel('Countries', fontsize=14, labelpad=10)
    plt.ylabel('Score assigned', fontsize=14, labelpad=10)

    plt.yticks(range(1, 11), fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)

    sns.despine(trim=True, offset=5)

    plt.tight_layout()
    plt.show()

#---5
def usaItalyGenreRadar(user_ratings, details_path='../data/details_clean.parquet'):

    details = pd.read_parquet(
        details_path,
        engine='pyarrow',
        columns=['mal_id', 'genres']
    )

    details.rename(columns={'mal_id': 'anime_id'}, inplace=True)

    usa_mask = user_ratings['location'] == 'United States'
    italy_mask = user_ratings['location'] == 'Italy'

    usa_anime_counts = user_ratings[usa_mask]['anime_id'].value_counts()
    italy_anime_counts = user_ratings[italy_mask]['anime_id'].value_counts()

    details['usa_votes'] = details['anime_id'].map(usa_anime_counts).fillna(0)
    details['italy_votes'] = details['anime_id'].map(italy_anime_counts).fillna(0)

    details_exploded = details.explode('genres')

    details_exploded['genre_list'] = details_exploded['genres'].astype(str).str.strip()

    invalid_tags = ['', 'nan', 'None', 'Unknown']

    details_exploded = details_exploded[
        details_exploded['genre_list'].notna() &
        ~details_exploded['genre_list'].isin(invalid_tags)
    ]

    genre_summary = details_exploded.groupby('genre_list').agg(
        usa_total=('usa_votes', 'sum'),
        italy_total=('italy_votes', 'sum')
    ).reset_index()

    genre_summary['USA Share (%)'] = (
        genre_summary['usa_total'] / genre_summary['usa_total'].sum()
    ) * 100

    genre_summary['Italy Share (%)'] = (
        genre_summary['italy_total'] / genre_summary['italy_total'].sum()
    ) * 100

    genre_summary['Total_Combined'] = (
        genre_summary['usa_total'] + genre_summary['italy_total']
    )

    top_genres = genre_summary.nlargest(8, 'Total_Combined')

    categories = top_genres['genre_list'].tolist()
    usa_values = top_genres['USA Share (%)'].tolist()
    italy_values = top_genres['Italy Share (%)'].tolist()

    categories.append(categories[0])
    usa_values.append(usa_values[0])
    italy_values.append(italy_values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=usa_values,
        theta=categories,
        fill='toself',
        name='United States (Top Market)',
        line_color='#3b82f6',
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=italy_values,
        theta=categories,
        fill='toself',
        name='Italy',
        line_color='#deff9a',
        fillcolor='rgba(222, 255, 154, 0.4)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(usa_values), max(italy_values)) + 2],
                ticksuffix="%"
            )
        ),
        title="Cultural Footprint: Gender Preferences (Italy vs. United States)",
        template='plotly_dark',
        font=dict(family="Arial, sans-serif", size=14),
        title_font=dict(size=18, family="Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=80, r=80, t=100, b=80)
    )

    fig.show()

#---6
def italyGlobalTopAnimeComparison(user_ratings, details_path='../data/details_clean.parquet'):

    italy_mask = user_ratings['location'] == 'Italy'
    italy_counts = user_ratings[italy_mask]['anime_id'].value_counts().reset_index()
    italy_counts.columns = ['anime_id', 'italy_votes']

    top_10_italy = italy_counts.head(10).copy()

    global_counts = user_ratings['anime_id'].value_counts().reset_index()
    global_counts.columns = ['anime_id', 'global_votes']

    top_10_global = global_counts.head(10).copy()

    unique_top_ids = list(
        set(top_10_italy['anime_id']).union(set(top_10_global['anime_id']))
    )

    details_titles = pd.read_parquet(
        details_path,
        engine='pyarrow',
        columns=['mal_id', 'title']
    )

    details_titles.rename(columns={'mal_id': 'anime_id'}, inplace=True)

    titles_subset = details_titles[
        details_titles['anime_id'].isin(unique_top_ids)
    ]

    final_subset = pd.merge(titles_subset, top_10_italy, on='anime_id', how='left').fillna(0)
    final_subset = pd.merge(final_subset, top_10_global, on='anime_id', how='left').fillna(0)

    final_subset['Italy Share (%)'] = (
        final_subset['italy_votes'] / top_10_italy['italy_votes'].sum()
    ) * 100

    final_subset['Global Share (%)'] = (
        final_subset['global_votes'] / top_10_global['global_votes'].sum()
    ) * 100

    final_subset.sort_values(by='italy_votes', ascending=True, inplace=True)

    melted_top = final_subset.melt(
        id_vars=['title'],
        value_vars=['Global Share (%)', 'Italy Share (%)'],
        var_name='Market',
        value_name='Share'
    )

    fig = px.bar(
        melted_top,
        y='title',
        x='Share',
        color='Market',
        barmode='group',
        orientation='h',
        title='Top Anime Comparison: Internal Market Share of Most Voted Series (%)',
        labels={
            'title': 'Anime Title',
            'Share': 'Relative Share within Market (%)'
        },
        color_discrete_map={
            'Global Share (%)': '#3b82f6',
            'Italy Share (%)': '#deff9a'
        },
        text_auto='.1f'
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_title='Relative Sample Share (%)',
        yaxis_title='',
        font=dict(family="Arial, sans-serif", size=12, color="#333333"),
        title_font=dict(size=18, family="Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=700,
        margin=dict(l=200, r=20, t=80, b=50)
    )

    fig.show()

#---7
def genreEvolutionOverTime(user_ratings, details_path='../data/details_clean.parquet'):

    year_column = 'year'

    details_time = pd.read_parquet(
        details_path,
        engine='pyarrow',
        columns=['mal_id', 'genres', year_column]
    )

    details_time.rename(columns={'mal_id': 'anime_id'}, inplace=True)

    details_time[year_column] = pd.to_numeric(details_time[year_column], errors='coerce')

    details_time = details_time[
        (details_time[year_column] >= 1990) &
        (details_time[year_column] <= 2023)
    ].copy()

    anime_vote_counts = user_ratings['anime_id'].value_counts().reset_index()
    anime_vote_counts.columns = ['anime_id', 'votes']

    time_genre_map = details_time.merge(anime_vote_counts, on='anime_id', how='inner')

    time_genre_exploded = time_genre_map.explode('genres')

    time_genre_exploded['genre_list'] = time_genre_exploded['genres'].astype(str).str.strip()

    invalid_tags = ['', 'nan', 'None', 'Unknown']

    time_genre_exploded = time_genre_exploded[
        time_genre_exploded['genre_list'].notna() &
        ~time_genre_exploded['genre_list'].isin(invalid_tags)
    ]

    top_5_genres = (
        time_genre_exploded.groupby('genre_list')['votes']
        .sum()
        .nlargest(5)
        .index
    )

    filtered_genres = time_genre_exploded[
        time_genre_exploded['genre_list'].isin(top_5_genres)
    ]

    final_evolution = (
        filtered_genres
        .groupby([year_column, 'genre_list'])['votes']
        .sum()
        .reset_index()
    )

    fig = px.area(
        final_evolution,
        x=year_column,
        y='votes',
        color='genre_list',
        title='Evolution of Anime Genres: The Impact of the 2013 Streaming Era',
        labels={
            year_column: 'Release Year',
            'votes': 'Total Cumulative Votes',
            'genre_list': 'Genre'
        },
        color_discrete_sequence=px.colors.qualitative.Vivid,
        category_orders={'genre_list': top_5_genres.tolist()}
    )

    fig.add_vline(
        x=2013,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=" 2013: Global Streaming Explosion",
        annotation_position="top left",
        annotation_font_color="red"
    )

    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        xaxis=dict(
            dtick=5,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            title='Engagement Volume (Total Votes)',
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        font=dict(family="Arial, sans-serif", size=12, color="#333333"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=20, t=100, b=50)
    )

    fig.show()

#---8
def productionShiftAnalysis(details_path='../data/details_clean.parquet'):


    colonna_anno = 'year'

    details_ind = pd.read_parquet(
        details_path,
        engine='pyarrow',
        columns=['mal_id', 'source', colonna_anno]
    )

    details_ind.rename(columns={'mal_id': 'anime_id'}, inplace=True)

    details_ind[colonna_anno] = pd.to_numeric(details_ind[colonna_anno], errors='coerce')

    details_ind = details_ind[
        (details_ind[colonna_anno] >= 1990) &
        (details_ind[colonna_anno] <= 2024)
    ].copy()

    details_ind['Era'] = np.where(
        details_ind[colonna_anno] < 2013,
        'Pre-Streaming (1990-2012)',
        'Streaming Era (2013-2024)'
    )

    details_ind['source_clean'] = (
        details_ind['source']
        .astype(str)
        .str.strip()
        .str.lower()
    )

    details_ind['Production_Type'] = np.where(
        details_ind['source_clean'] == 'original',
        'Original Work',
        'Adaptation (Manga/LN/Game/Other)'
    )

    structure_counts = (
        details_ind
        .groupby(['Era', 'Production_Type'])
        .size()
        .reset_index(name='count')
    )

    era_totals = (
        structure_counts
        .groupby('Era')['count']
        .sum()
        .reset_index(name='era_total')
    )

    structure_merged = structure_counts.merge(era_totals, on='Era')

    structure_merged['Share (%)'] = (
        structure_merged['count'] / structure_merged['era_total']
    ) * 100

    fig = px.bar(
        structure_merged,
        x='Production_Type',
        y='Share (%)',
        color='Era',
        barmode='group',
        title='Anime Production Strategy Shift: Original Works vs. Adaptations',
        labels={
            'Production_Type': 'Type of Production Source',
            'Share (%)': 'Proportion of Total Production (%)'
        },
        color_discrete_map={
            'Pre-Streaming (1990-2012)': '#3b82f6',
            'Streaming Era (2013-2024)': '#deff9a'
        },
        text_auto='.1f'
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_title='Production Source Category',
        yaxis_title='Percentage of Total Framework (%)',
        font=dict(family="Arial, sans-serif", size=12, color="#333333"),
        title_font=dict(size=18, family="Arial, sans-serif"),
        legend=dict(
            title='Production Era',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=20, t=100, b=50)
    )

    fig.show()

#--9
def platformPortfolioAnalysis(details_path='../data/details_clean.parquet'):

    details_plat = pd.read_parquet(
        details_path,
        engine='pyarrow',
        columns=['mal_id', 'title', 'score', 'members', 'licensors']
    )

    details_plat.rename(columns={'mal_id': 'anime_id'}, inplace=True)

    details_plat.dropna(
        subset=['score', 'members', 'licensors'],
        inplace=True
    )

    details_exploded = details_plat.explode('licensors')

    details_exploded['licensor_clean'] = (
        details_exploded['licensors']
        .astype(str)
        .str.strip()
    )

    target_platforms = [
        'Crunchyroll',
        'Funimation',
        'Sentai Filmworks',
        'VIZ Media'
    ]

    catalog_data = details_exploded[
        details_exploded['licensor_clean'].isin(target_platforms)
    ].copy()

    catalog_data.drop_duplicates(
        subset=['anime_id', 'licensor_clean'],
        inplace=True
    )

    fig = px.scatter(
        catalog_data,
        x='members',
        y='score',
        color='licensor_clean',
        hover_name='title',
        hover_data={
            'licensor_clean': False,
            'members': True,
            'score': True
        },
        title='Portfolio Analysis: Quality (Score) vs. Popularity (Members) by Platform',
        labels={
            'members': 'Total Popularity/Engagement (log scale)',
            'score': 'Score',
            'licensor_clean': 'Platform'
        },
        opacity=0.7,
        log_x=True,
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[2, 10]
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#333333"
        ),
        title_font=dict(
            size=18,
            family="Arial, sans-serif"
        ),
        legend=dict(
            title='Piattaforma',
            itemsizing='constant'
        ),
        margin=dict(
            l=50,
            r=20,
            t=80,
            b=50
        )
    )

    fig.show()