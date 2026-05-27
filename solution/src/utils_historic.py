"""
Generic utility functions
"""
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

outsourcing_keywords = [
    # Coreani
    'KIM', 'LEE', 'PARK', 'CHOI', 'JUNG', 'KANG', 'CHO', 'YOON', 'JANG', 'LIM', 'HAN', 'BAEK',
    # Cinesi
    'WANG', 'LI', 'ZHANG', 'LIU', 'CHEN', 'YANG', 'HUANG', 'ZHAO', 'WU', 'ZHOU', 'LIN',
    # Vietnamiti
    'NGUYEN', 'TRAN', 'LE', 'PHAM', 'HOANG',
    # Keyword Corporate (Spesso interi studi sono creditati al posto dei singoli)
    'STUDIO', 'ANIMATION', 'ENTERTAINMENT', 'CO.', 'LTD', 'INC', 'WORKS', 'DR MOVIE'
]

def is_likely_outsourced_advanced(row):
    # Gestione sicura dei nomi
    name = str(row['name']).upper() if pd.notna(row['name']) else ""
    # Usiamo get() per evitare errori se la colonna non esiste
    given_name = row.get('given_name', None)

    # Escludiamo il placeholder che avevamo creato per i professionisti ignoti
    if name == "UNKNOWN PROFESSIONAL":
        return False

    # CONDIZIONE 1: Match con la lista espansa
    if any(keyword in name for keyword in outsourcing_keywords):
        return True

    # CONDIZIONE 2: L'intuizione del 'given_name' vuoto
    if pd.isna(given_name) or str(given_name).strip() == "":
        return True

    return False


asian_kw = ['KIM','LEE','PARK','CHOI','JUNG','KANG','CHO','YOON','JANG','LIM','HAN','BAEK',
            'WANG','LI','ZHANG','LIU','CHEN','YANG','HUANG','ZHAO','WU','ZHOU','LIN',
            'NGUYEN','TRAN','LE','PHAM','HOANG']
corp_kw = ['STUDIO','ANIMATION','ENTERTAINMENT','CO.','LTD','INC','WORKS','DR MOVIE']

def classify_origin(row):
    if pd.notna(row.get('location_country')):
        return 'Japan' if row['is_japan'] else 'foreign (location)'
    name_upper = str(row.get('name','')).upper()
    given = row.get('given_name')
    if pd.isna(given) or str(given).strip() == '':
        return 'foreign (no given_name)'
    if any(kw in name_upper for kw in asian_kw + corp_kw):
        return 'foreign (surname/keyword)'
    return 'Japan (exclusion)'

region_map = {
    'South Korea': 'Corea del Sud', 'Korea': 'Corea del Sud',
    'China': 'Cina', 'Taiwan': 'Taiwan',
    'Philippines': 'Filippine', 'Vietnam': 'Vietnam',
    'India': 'India', 'Indonesia': 'Sud-est Asiatico', 'Thailand': 'Sud-est Asiatico',
    'USA': 'Nord America', 'Canada': 'Nord America',
    'UK': 'Europa', 'France': 'Europa', 'Germany': 'Europa',
    'Italy': 'Europa', 'Spain': 'Europa',
    'Japan': 'Giappone',
    'Brazil': 'Sud America', 'Mexico': 'Sud America',
    'Australia': 'Oceania', 'South Africa': 'Africa'
}

def get_region(row):
    if pd.notna(row.get('country')):
        return region_map.get(row['country'], 'other')
    if 'foreign' in str(row.get('origin','')):
        return 'foreign (not located)'
    return 'Japan'

def assign_era(year):
    if year < 1990: return 'A-Era Tradizionale\n(Traditional Era)\n1960-1989'
    elif year < 2010: return 'B-Cavalieri 2000s\n(2000s Knights)\n2000-2009'
    else: return 'C-Nuova Generazione\n(New Generation)\n2010-2025'

def get_top_with_others(df, era_label, n=8):
    era_data = df[df['era'] == era_label].sort_values('count', ascending=False)
    top = era_data.head(n).copy()
    others_count = era_data.iloc[n:]['count'].sum() if len(era_data) > n else 0
    if others_count > 0:
        others_total = era_data.iloc[n:]['total'].iloc[0] if len(era_data) > n else top['total'].iloc[0]
        others_share = (others_count / top['total'].iloc[0] * 100).round(1)
        others_row = pd.DataFrame([{'era': era_label, 'studios': 'Altri (Others)', 'count': others_count, 'total': top['total'].iloc[0], 'share': others_share}])
        return pd.concat([top, others_row], ignore_index=True)
    return top

#---------------------------------------------------------------------------------------------------------------------
# Graph functions

#Phase1
def plot_pioneers_genres(details):
    # First years genres analysis (es. until 1960s)
    origini_genres = details[details['year'] < 1960].copy()
    
    # Exploding lists to count each genre
    genres_exploded = origini_genres.explode('genres')
    
    # Counting genres and creating clean mini-dataset
    genre_counts = genres_exploded['genres'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Counting']
    
    plt.figure(figsize=(10, 5))
    
    sns.barplot(
        data=genre_counts,
        x='Counting',     
        y='Genre',
        hue='Genre',
        palette='copper_r',
        legend=False
    )
    
    plt.title('Main genres in pioneers era (before-1960)')
    plt.xlabel('Number of works')
    plt.ylabel('Genre')
    plt.savefig("barlot_pionieri.png", dpi=300, bbox_inches='tight')
    plt.show()


def distribution_format_evolution(details):
    # Distribution format evolution across decades
    df_type_decade = details.copy()
    df_type_decade['decade'] = (df_type_decade['year'] // 10) * 10
    type_pivot = pd.crosstab(df_type_decade['decade'], df_type_decade['type'])
    type_pivot = type_pivot[type_pivot.index >= 1950]
    ax = type_pivot.plot(
        kind='bar', stacked=True, figsize=(14, 7),
        colormap='tab10', edgecolor='white', linewidth=0.5
    )
    plt.title('Evolution of Release Formats by Decade', fontsize=15, pad=15)
    plt.xlabel('Decade', fontsize=12)
    plt.ylabel('number of works', fontsize=12)
    plt.legend(title='Format', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("format_per_decade.png", dpi=300, bbox_inches='tight')
    plt.show()


def production_seasonality(details):
    # Production seasonality: month × decade heatmap
    df_season = details.dropna(subset=['start_date']).copy()
    df_season['month'] = df_season['start_date'].dt.month
    df_season['decade'] = (df_season['year'] // 10) * 10
    
    # Filtering historic decades to see changes
    df_season_filtered = df_season[df_season['decade'] >= 1960]
    heatmap_data = pd.crosstab(df_season_filtered['month'], df_season_filtered['decade'])
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data, annot=True, fmt='d',
        cmap='YlOrRd', linewidths=0.5,
        cbar_kws={'label': 'Number of new works'}
    )
    plt.title('Anime Release Seasonality - Month × Decade', fontsize=14, pad=15)
    plt.xlabel('decade', fontsize=12)
    plt.ylabel('month', fontsize=12)
    plt.tight_layout()
    plt.savefig("seasonal_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


#Phase2
def genres_market_share_evolution(details):
    # Preparation of global dataset by exploding genres over decades
    df_genres_time = details.explode('genres')
    
    # Calculating share percentage for each genre in every decade
    genre_decade_counts = df_genres_time.groupby(['decade', 'genres']).size().unstack(fill_value=0)
    genre_decade_perc = genre_decade_counts.div(genre_decade_counts.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(12, 6))
    target_genres = ['Action', 'Comedy', 'Sci-Fi', 'Fantasy']
    
    for g in target_genres:
        if g in genre_decade_perc.columns:
            plt.plot(genre_decade_perc.index, genre_decade_perc[g], marker='o', linewidth=3, label=g)
    
    plt.title('Genres market share evolution (1960 - 2020)', fontsize=15, pad=15)
    plt.xlabel('Decades', fontsize=12)
    plt.ylabel('% over amount of producted genres', fontsize=12)
    plt.xlim(1960, 2020)
    plt.ylim(0, 30)
    plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plot_fase2.png", dpi=300, bbox_inches='tight')
    plt.show()


def anime_genres_market_composition(details):
    # Preparation of global dataset by exploding genres over decades
    df_genres_time = details.explode('genres')
    
    # Calculating share percentage for each genre in every decade
    genre_decade_counts = df_genres_time.groupby(['decade', 'genres']).size().unstack(fill_value=0)
    genre_decade_perc = genre_decade_counts.div(genre_decade_counts.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(12, 6))
    
    target_genres = ['Action', 'Comedy', 'Sci-Fi', 'Fantasy']
    valid_genres = [g for g in target_genres if g in genre_decade_perc.columns]
    colours = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    
    # Area graphs from dataFrame
    ax = genre_decade_perc[valid_genres].plot.area(
        color=colours,
        alpha=0.85,         
        figsize=(12, 6),
        linewidth=0          
    )
    
    plt.title('Anime genres market composition (1960 - 2020)', fontsize=15, pad=15)
    plt.xlabel('Decade', fontsize=12)
    plt.ylabel('% over total', fontsize=12)
    plt.xlim(1960, 2020)
    plt.ylim(bottom=0)
    plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("stacked_area_chart_fase2.png", dpi=300, bbox_inches='tight')
    plt.show()


def score_distribution_by_genre_decade(details):
    # Boxplot: Score distribution by genre, faceted by decade (1960-2020)
    df_score_genre = details.explode('genres').copy()
    df_score_genre = df_score_genre.dropna(subset=['score'])
    
    top_genres = df_score_genre['genres'].value_counts().nlargest(8).index.tolist()
    df_score_genre = df_score_genre[df_score_genre['genres'].isin(top_genres)]
    
    df_score_genre['decade'] = (df_score_genre['year'] // 10) * 10
    
    # Rows removal with empty year 
    df_score_genre = df_score_genre.dropna(subset=['decade'])
    # Conversion of column into standard numeric type
    df_score_genre['decade'] = df_score_genre['decade'].astype('int64')
    
    df_score_genre = df_score_genre[df_score_genre['decade'].between(1960, 2020)]
    
    g = sns.catplot(
        data=df_score_genre,
        x='decade', y='score', hue='decade',
        col='genres', col_wrap=4,
        kind='box', height=4, aspect=0.8,
        sharey=True, palette='viridis',
        fliersize=2,
        legend=False
    )
    
    g.set_axis_labels('Decade', 'Score')
    g.set_titles('{col_name}')
    g.figure.suptitle('Score Distribution by Genre and Decade', y=1.02, fontsize=15)
    plt.tight_layout()
    plt.savefig("boxplot_score_genere_decade.png", dpi=300, bbox_inches='tight')
    plt.show()


def target_dem_evolution_decade(details):
    # Target demographics evolution by decade
    df_demo = details.explode('demographics').dropna(subset=['demographics'])
    df_demo['decade'] = (df_demo['year'] // 10) * 10
    df_demo = df_demo[df_demo['decade'].between(1960, 2020)]
    demo_counts = df_demo.groupby(['decade', 'demographics']).size().unstack(fill_value=0)
    demo_counts.plot(
        kind='area', stacked=True, figsize=(12, 6),
        colormap='Set1', alpha=0.8
    )
    plt.title('Target Demographics Evolution by Decade', fontsize=15, pad=15)
    plt.xlabel('Decades', fontsize=12)
    plt.ylabel('Number of works', fontsize=12)
    plt.legend(title='Demographic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(1960, 2020)
    plt.tight_layout()
    plt.savefig("demographics_evolution.png", dpi=300, bbox_inches='tight')
    plt.show()


def production_volume_average_yearly_score(details):
    # Correlation between production volume and average yearly score
    df_vol_qual = details.groupby('year').agg(
        avg_score=('score', 'mean'),
        anime_count=('mal_id', 'count')
    ).reset_index()
    df_vol_qual = df_vol_qual[df_vol_qual['year'].between(1960, 2024)]
    fig, ax1 = plt.subplots(figsize=(14, 6))
    color1, color2 = '#4C72B0', '#DD8452'
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of producted anime', color=color1, fontsize=12)
    ax1.bar(df_vol_qual['year'], df_vol_qual['anime_count'], color=color1, alpha=0.3, label='Production volume')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean score', color=color2, fontsize=12)
    ax2.plot(df_vol_qual['year'], df_vol_qual['avg_score'], color=color2, linewidth=3, marker='o', markersize=4, label='Score Medio')
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title('Production Volume vs Average Yearly Score', fontsize=15, pad=15)
    fig.tight_layout()
    plt.savefig("volume_vs_quality.png", dpi=300, bbox_inches='tight')
    plt.show()


def mean_staff_evolution_per_works(df_industrial):
    plt.figure(figsize=(10, 6))
    # Data copying & decade conversion into standard type to avoid Seaborn bugs
    df_plot = df_industrial[(df_industrial['decade'] >= 1960) & (df_industrial['decade'] <= 2020)].copy()
    df_plot['decade'] = df_plot['decade'].astype('int64')
    
    sns.barplot(
        data=df_plot,
        x='decade',
        y='staff_size',
        hue='decade',
        palette='crest',
        errorbar=None,
        legend=False
    )
    
    plt.title('Mean staff evolution per works', fontsize=14)
    plt.xlabel('Decade')
    plt.ylabel('Mean of professionists in techincal cast')
    plt.savefig("barplot_fase3.png", dpi=300, bbox_inches='tight')
    plt.show()


def anime_globalization_outsourcing(person_details, person_anime_works, details):
    person_details['is_outsourced'] = person_details.apply(is_likely_outsourced_advanced, axis=1)
    df_staff_full = pd.merge(person_anime_works, person_details[['person_mal_id', 'is_outsourced']], on='person_mal_id', how='left')
    df_outsourcing = pd.merge(df_staff_full, details[['mal_id', 'year']], left_on='anime_mal_id', right_on='mal_id', how='inner')
    
    yearly_trend = df_outsourcing.groupby(['year', 'is_outsourced'])['person_mal_id'].count().reset_index(name='staff_count')
    yearly_trend['Origine'] = yearly_trend['is_outsourced'].map({True: 'Estero / Entità Corporate', False: 'Giappone (Domestico)'})
    yearly_trend = yearly_trend[(yearly_trend['year'] >= 1960) & (yearly_trend['year'] <= 2024)]
    
    fig = px.area(
        yearly_trend,
        x="year",
        y="staff_count",
        color="Origine",
        color_discrete_map={
            'Japan': '#005088',
            'foreign / Corporate': '#11CAA0'
        },
        title="Anime globalization: Outsourcing & external studios explosion (1960-2024)",
        labels={"year": "publication year", "staff_count": "Workforce employed"},
        template="plotly_white"
    )
    
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            title_font_family="Arial",
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    fig.show()


def globalization_animation_labor(pivot):
    # absolute composition & outsourcing percentage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    origin_colors = {
        'Japan': '#005088',
        'Japan (exclusion)': '#4A90D9',
        'foreign (location)': '#11CAA0',
        'foreign (no given_name)': '#88D8B0',
        'foreign (surname/keyword)': '#FFA15A'
    }
    order = [c for c in origin_colors if c in pivot.columns]
    
    pivot[order].plot(kind='area', stacked=True, ax=ax1,
                      color=[origin_colors[c] for c in order],
                      alpha=0.85, linewidth=0)
    ax1.set_title('Animation Workforce Composition', fontsize=14)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Counting Animation Staff' , fontsize=12)
    ax1.set_xlim(1960, 2024)
    ax1.legend(title='Origin', bbox_to_anchor=(1.05,1), loc='upper left')
    
    pct = pivot[pivot.index >= 1980]
    ax2.fill_between(pct.index, 0, pct['% Outsourcing'], color='#11CAA0', alpha=0.6, label='% Outsourcing')
    ax2.plot(pct.index, pct['% Outsourcing'], color='#0D8A6E', linewidth=2.5, marker='o', markersize=4)
    ax2.set_title('Outsourced Animation %', fontsize=14)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('% Outsourcing over amount of animators', fontsize=12)
    ax2.set_xlim(1980, 2024)
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.4, label='limit 50%')
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.suptitle('Globalization of Animation Labor', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("outsourcing_refined_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def international_animation_workforce_regions(person_details, animator_works, details):
    # Breakdown of the international component by geographic macro-region
    person_details['country'] = person_details['relevant_location'].dropna().str.split(', ').str[-1]
    
    person_details['region'] = person_details.apply(get_region, axis=1)
    df_reg = pd.merge(animator_works, person_details[['person_mal_id','region']], on='person_mal_id', how='left')
    df_reg = pd.merge(df_reg, details[['mal_id','year']], left_on='anime_mal_id', right_on='mal_id', how='inner')
    region_trend = df_reg.groupby(['year','region']).size().reset_index(name='count')
    region_pivot = region_trend.pivot_table(index='year', columns='region', values='count', fill_value=0)
    
    # Only outsource regions (excluding Japan and not located ones)
    outsource_regs = [r for r in region_pivot.columns
                      if r not in ['Japan', 'foreign (not located)']]
    if outsource_regs:
        region_pivot[outsource_regs].plot(
            kind='area', stacked=True, figsize=(14, 7),
            colormap='tab10', alpha=0.8
        )
        plt.title('International Animation Workforce by Region', fontsize=15, pad=15)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Counting foreign animators', fontsize=12)
        plt.xlim(1980, 2024)
        plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("outsourcing_by_region.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No foreign regions identified using available data")


def rising_decline_animation_empires(df_studios_exploded, top_studios):
    # Filtering only top studios data
    df_top_studios = df_studios_exploded[df_studios_exploded['studios'].isin(top_studios)].copy()
    
    # Calculating decade
    df_top_studios['decade'] = (df_top_studios['year'] // 10) * 10
    
    # --- Removal null values before conversion ---
    df_top_studios = df_top_studios.dropna(subset=['decade'])
    
    # From Seaborn bug
    df_top_studios['decade'] = df_top_studios['decade'].astype('int64')
    
    studio_trends = df_top_studios.groupby(['decade', 'studios']).size().reset_index(name='anime_count')
    
    # Filtering decades to focus on Tv era and beyond (1960-2020)
    studio_trends = studio_trends[studio_trends['decade'].between(1960, 2020)]
    
    plt.figure(figsize=(14, 7))
    
    # lineplot with marker to highlight decades
    sns.lineplot(
        data=studio_trends,
        x='decade',
        y='anime_count',
        hue='studios',
        marker='o',
        markersize=10,
        linewidth=3,
        palette='tab10'
    )
    
    plt.title("Rising and decline of Animation empires (1960 - 2020)", fontsize=16, pad=20)
    plt.xlabel("Decade", fontsize=12)
    plt.ylabel("Number of work published", fontsize=12)
    plt.legend(title='Animation studios', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("ascesa_studi_animazione.png", dpi=300, bbox_inches='tight')
    plt.show()


def market_share_by_era(details):
    # Market share of top studios per era
    details_st = details.copy()
    details_st['decade'] = (details_st['year'] // 10) * 10
    studios_df = details_st.explode('studios').reset_index(drop=True)
    studios_df = studios_df[studios_df['studios'].notna() & (studios_df['studios'] != '')]
    # Exlcuding non-japanese studios
    non_jp = ['Shanghai Animation Film Studio', 'AQUA ARIS', 'T-Rex', 'Ruo Hong Culture', 'HAOLINERS', 'CBC']
    studios_jp = studios_df[~studios_df['studios'].isin(non_jp)]
    # Top 12 studi giapponesi per volume totale
    top12 = studios_jp['studios'].value_counts().head(12).index.tolist()
    studios_top = studios_jp[studios_jp['studios'].isin(top12)].copy()
    studios_top['era'] = studios_top['year'].apply(assign_era)
    
    # Calculating market share: how many anime each studio product per era?
    era_counts = studios_top.groupby(['era', 'studios']).size().reset_index(name='count')
    era_totals = era_counts.groupby('era')['count'].sum().reset_index(name='total')
    era_counts = era_counts.merge(era_totals, on='era')
    era_counts['share'] = (era_counts['count'] / era_counts['total'] * 100).round(1)
    
    eras_list = era_counts['era'].unique()
    chart_data = pd.concat([get_top_with_others(era_counts, e) for e in eras_list], ignore_index=True)
    
    studio_palette = {
        'Toei Animation': '#E63946', 'Sunrise': '#457B9D', 'J.C.Staff': '#A8DADC',
        'Madhouse': '#F4A261', 'Production I.G': '#264653', 'A-1 Pictures': '#2A9D8F',
        'TMS Entertainment': '#E9C46A', 'Studio Deen': '#D4A373', 'OLM': '#6D6875',
        'Studio Pierrot': '#FFB703', 'Shin-Ei Animation': '#FB8500', 'Nippon Animation': '#219EBC',
        'AIC': '#8ECAE6', 'Tatsunoko Production': '#8338EC', 'DLE': '#FF006E',
        'Altri (Others)': '#CCCCCC'
    }
    
    # Horizontal bar chart group by era
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("The Studio Battle: Market Share by Era",
                 fontsize=16, y=1.02, fontweight='bold')
    for idx, era_label in enumerate(eras_list):
        ax = axes[idx]
        data = chart_data[chart_data['era'] == era_label].sort_values('share')
        colors = [studio_palette.get(s, '#CCCCCC') for s in data['studios']]
    
        bars = ax.barh(data['studios'], data['share'], color=colors, edgecolor='white', height=0.7)
        ax.bar_label(bars, labels=[f"{v}%" for v in data['share']], padding=3, fontsize=9)
    
        ax.set_title(era_label.replace('\\n', '\n'), fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Market share (%)', fontsize=11)
        ax.set_xlim(0, 100)
        ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    plt.savefig("studio_market_share.png", dpi=300, bbox_inches='tight')
    plt.show()


def volume_vs_quality_across_eras(details):
    # Market share of top studios per era
    details_st = details.copy()
    details_st['decade'] = (details_st['year'] // 10) * 10
    studios_df = details_st.explode('studios').reset_index(drop=True)
    studios_df = studios_df[studios_df['studios'].notna() & (studios_df['studios'] != '')]
    # Exlcuding non-japanese studios
    non_jp = ['Shanghai Animation Film Studio', 'AQUA ARIS', 'T-Rex', 'Ruo Hong Culture', 'HAOLINERS', 'CBC']
    studios_jp = studios_df[~studios_df['studios'].isin(non_jp)]
    # Top 12 studi giapponesi per volume totale
    top12 = studios_jp['studios'].value_counts().head(12).index.tolist()
    studios_top = studios_jp[studios_jp['studios'].isin(top12)].copy()
    studios_top['era'] = studios_top['year'].apply(assign_era)

    # Volume vs Quality per studio with temporal evolution
    era_defs = {'Traditional\n(1960-1989)': (1960, 1989), '2000s': (2000, 2009), 'New Gen.\\n(2010-2025)': (2010, 2025)}
    studio_era_stats = []
    for era_name, (y1, y2) in era_defs.items():
        subset = studios_top[studios_top['year'].between(y1, y2)]
        stats = subset.groupby('studios').agg(
            count=('mal_id', 'count'),
            avg_score=('score', 'mean'),
            med_score=('score', 'median')
        ).reset_index()
        stats['era'] = era_name
        studio_era_stats.append(stats)
    df_scatter = pd.concat(studio_era_stats, ignore_index=True)
    
    # Filtering studios that made at least 5 works per era to get statistic strenght
    df_scatter = df_scatter[df_scatter['count'] >= 5]
    # Scatter plot
    fig, ax = plt.subplots(figsize=(14, 9))
    era_colors = {'Traditional\n(1960-1989)': '#457B9D', '2000s': '#E9C46A', 'New Gen.\n(2010-2025)': '#E63946'}
    era_markers = {'Traditional\n(1960-1989)': 'o', '2000s': 's', 'New Gen.\n(2010-2025)': 'D'}
    for era_name in df_scatter['era'].unique():
        data = df_scatter[df_scatter['era'] == era_name]
        ax.scatter(data['count'], data['avg_score'],
                   c=era_colors.get(era_name, '#999'),
                   marker=era_markers.get(era_name, 'o'),
                   s=data['count']*2 + 50, alpha=0.7,
                   edgecolors='white', linewidth=1,
                   label=era_name)
    # Notes about most important studios for each era
    for _, row in df_scatter.iterrows():
        if row['count'] >= 50:  # High productive studios
            ax.annotate(row['studios'],
                        (row['count'], row['avg_score']),
                        fontsize=7.5, alpha=0.8,
                        xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Productive volum (no. works)', fontsize=12)
    ax.set_ylabel('Mean score', fontsize=12)
    ax.set_title('Volume vs Quality: Studio Evolution Across Eras', fontsize=15, pad=15, fontweight='bold')
    ax.legend(title='Historic era', loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    # Reference line about global mean score
    global_avg = details['score'].mean()
    ax.axhline(y=global_avg, color='gray', linestyle='--', alpha=0.5,
               label=f'Score medio globale ({global_avg:.2f})')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("studio_quality_vs_volume4.png", dpi=300, bbox_inches='tight')
    plt.show()

def format_specialization_heatmap(details):
    # Market share of top studios per era
    details_st = details.copy()
    details_st['decade'] = (details_st['year'] // 10) * 10
    studios_df = details_st.explode('studios').reset_index(drop=True)
    studios_df = studios_df[studios_df['studios'].notna() & (studios_df['studios'] != '')]
    # Exlcuding non-japanese studios
    non_jp = ['Shanghai Animation Film Studio', 'AQUA ARIS', 'T-Rex', 'Ruo Hong Culture', 'HAOLINERS', 'CBC']
    studios_jp = studios_df[~studios_df['studios'].isin(non_jp)]
    # Top 12 studi giapponesi per volume totale
    top12 = studios_jp['studios'].value_counts().head(12).index.tolist()
    studios_top = studios_jp[studios_jp['studios'].isin(top12)].copy()
    studios_top['era'] = studios_top['year'].apply(assign_era)

    # Format Specialization - Studio x Format Heatmap
    format_pivot = pd.crosstab(
        studios_top['studios'],
        studios_top['type']
    )
    # Ordering by volumes
    format_pivot = format_pivot.loc[format_pivot.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(
        format_pivot,
        annot=True, fmt='d',
        cmap='YlOrRd',
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Number of works', 'shrink': 0.8},
        ax=ax
    )
    ax.set_title("Studio Format Specialization - # of Works",
                 fontsize=15, pad=15, fontweight='bold')
    ax.set_xlabel('Format', fontsize=12)
    ax.set_ylabel('Studio', fontsize=12)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("studio_format_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

