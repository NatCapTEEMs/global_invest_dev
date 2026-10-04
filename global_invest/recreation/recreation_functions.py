"""Recreation/tourism science: site quality, banded travel-cost flows, three value channels.

The method is the author's zonal travel-cost rewrite (m-braaksma/gep_recreation through
540f7af); the constants below ARE that method, so they live in code -- a change is a reviewed
commit, not an input/-copy edit.

Method in one pass: LULC class shares + protected-area share -> a 0-3 environment class;
urban share + distance-to-road -> a 1-5 accessibility class; the two cross into a 1-9 site
rank, of which class 9 is a high-quality site, dissolved into discrete sites by connected
components. Demand is a per-capita trip budget (participation a) split over distance bands by
an exponential decay (rate b), both read per country from the demand-parameter table; each
band's fixed budget divides evenly over the sites its ring reaches, the rest is unmet, and
realized visits price at the origin country's per-km fuel cost over the round trip to the band
midpoint. Two further channels add on: overnight stays priced at the hotel-price surface, and
arrivals by origin region priced at the staged region-to-region airfare matrix.

Raster ops are pure array functions (unit-testable) wrapped into pygeoprocessing closures by
the task layer; the band maths is closed form (cdf_exp), so no numerical integration anywhere.
"""

import numpy as np
import pandas as pd

# --- Method constants (the author's method; see module docstring) ---
RECREATION_LULC_CLASSES = ('cropland', 'forest', 'grassland', 'othernat', 'urban', 'water')
RECREATION_LULC_SCORES = {'cropland': 0.4, 'forest': 1.0, 'grassland': 0.7,
                          'othernat': 0.85, 'urban': 0.05, 'water': 1.0}
RECREATION_PA_SCORE = 1.0
RECREATION_ENV_BINS = (1.5, 1.0, 0.5, 0.0)          # combined score > bin -> class 3, 2, 1, 0
RECREATION_URBAN_BINS = (0.9, 0.5, 0.1, 0.001, 0.0)  # urban share > bin -> class 4, 3, 2, 1, 0
RECREATION_ROAD_BINS = (50.0, 20.0, 5.0, 2.0, 0.0)   # road distance > bin -> class 0, 1, 2, 3, 4
RECREATION_URBAN_ROAD_MATRIX = np.array([            # rows: urban class, cols: road class -> 1-5
    [1, 1, 2, 3, 4],
    [1, 1, 2, 3, 4],
    [2, 2, 2, 4, 5],
    [3, 3, 4, 5, 5],
    [3, 4, 4, 5, 5]])
RECREATION_SITE_MATRIX = np.array([                  # rows: accessibility 1-5, cols: env 0-3 -> 1-9
    [1, 1, 4, 7],
    [1, 4, 4, 7],
    [2, 2, 8, 8],
    [3, 5, 5, 9],
    [3, 6, 6, 9]])
RECREATION_HQ_SITE_CLASS = 9
RECREATION_FUEL_COST_COL = 'gasoline_cost_usd_per_km_2019_gppdata'

INT_NDV = -1
FLOAT_NDV = -9999.0


# --- Pure array ops (the science; wrapped into raster_calculator closures below) ---
def environment_class_array(crop, forest, grass, othernat, urban, water, pa):
    """Weighted LULC-share composite + protected-area share -> environment class 0-3."""
    combined = (RECREATION_LULC_SCORES['cropland'] * crop
                + RECREATION_LULC_SCORES['forest'] * forest
                + RECREATION_LULC_SCORES['grassland'] * grass
                + RECREATION_LULC_SCORES['othernat'] * othernat
                + RECREATION_LULC_SCORES['urban'] * urban
                + RECREATION_LULC_SCORES['water'] * water
                + RECREATION_PA_SCORE * pa)
    return np.select(
        condlist=[combined > RECREATION_ENV_BINS[0], combined > RECREATION_ENV_BINS[1],
                  combined > RECREATION_ENV_BINS[2], combined >= RECREATION_ENV_BINS[3]],
        choicelist=[3, 2, 1, 0], default=0)


def accessibility_class_array(urban, roads):
    """Urban share x distance-to-road -> accessibility class 1-5 via the urban/road matrix."""
    urban_class = np.select(
        condlist=[urban > RECREATION_URBAN_BINS[0], urban > RECREATION_URBAN_BINS[1],
                  urban > RECREATION_URBAN_BINS[2], urban > RECREATION_URBAN_BINS[3],
                  urban >= RECREATION_URBAN_BINS[4]],
        choicelist=[4, 3, 2, 1, 0], default=0)
    road_class = np.select(
        condlist=[roads > RECREATION_ROAD_BINS[0], roads > RECREATION_ROAD_BINS[1],
                  roads > RECREATION_ROAD_BINS[2], roads > RECREATION_ROAD_BINS[3],
                  roads >= RECREATION_ROAD_BINS[4]],
        choicelist=[0, 1, 2, 3, 4], default=0)
    return RECREATION_URBAN_ROAD_MATRIX[urban_class, road_class].astype(np.int32)


def site_rank_array(acc_class, env_class):
    """Accessibility class 1-5 x environment class 0-3 -> site rank 1-9."""
    return RECREATION_SITE_MATRIX[acc_class - 1, env_class].astype(np.int16)


# --- Banded travel-cost demand (the author's zonal rewrite) ---
def cdf_exp(d_km, b):
    """CDF of the normalized exponential density f(d) = b exp(-b d); closed form."""
    return 1.0 - np.exp(-b * d_km)


def compute_group_bands(pixel_size_km, max_distance_km=50.0, max_bands=8):
    """The fixed-cutoff distance bands, grid-aware, exactly as the source computes them.

    The cutoff is the same for every parameter group (a farthest reasonable day trip); the
    band count is floor(cutoff / pixel), clamped to [1, max_bands], so no band is narrower
    than the grid resolves. The decay rate b shapes how demand SPLITS over these bands
    (cdf_exp), never where it stops.

    Returns:
        (cutoff_km, edges): edges is the list of n_bands + 1 km values from 0 to the cutoff.
    """
    cutoff_km = max_distance_km
    n_bands = int(np.clip(np.floor(cutoff_km / pixel_size_km), 1, max_bands))
    band_width = cutoff_km / n_bands
    edges = [i * band_width for i in range(n_bands + 1)]
    return cutoff_km, edges


def ring_kernel_array(r_lo, r_hi, pixel_size_km):
    """Binary ring indicator over pixel offsets: 1 where the offset distance falls in
    (r_lo, r_hi], or [0, r_hi] for the first band. One kernel serves both the site count
    within the band and the even spread of the band's fixed budget."""
    radius_px = int(np.ceil(r_hi / pixel_size_km))
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    d_km = np.sqrt(xx ** 2 + yy ** 2) * pixel_size_km
    within = (d_km <= r_hi) if r_lo == 0 else ((d_km > r_lo) & (d_km <= r_hi))
    return within.astype(np.float32)


def validate_recreation_params(df):
    """The demand-parameter table as the two lookups the flow engine needs.

    Args:
        df (pd.DataFrame): iso3_r250_id, iso3_r250_name, param_group, participation_param,
            distance_param.

    Returns:
        (country_to_group_map, group_params): id -> group, and group -> {'a', 'b'}.

    Raises:
        ValueError: on missing columns, or a param_group whose countries disagree on the
            parameter values -- that disagreement means the grouping and the parameters have
            different ideas of what a group is.
    """
    required = ['iso3_r250_id', 'iso3_r250_name', 'param_group',
                'participation_param', 'distance_param']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Parameter table missing required columns: {missing_cols}')
    check = df.groupby('param_group')[['participation_param', 'distance_param']].nunique()
    bad = check[(check['participation_param'] > 1) | (check['distance_param'] > 1)]
    if not bad.empty:
        raise ValueError(
            'param_group(s) with inconsistent participation_param/distance_param across '
            f'countries: {list(bad.index)}. Every country sharing a param_group must have '
            'identical parameter values.')
    country_to_group_map = dict(zip(df['iso3_r250_id'].astype(int), df['param_group'].astype(int)))
    group_params = {}
    for gid, sub in df.groupby('param_group'):
        row = sub.iloc[0]
        group_params[int(gid)] = {'a': float(row['participation_param']),
                                  'b': float(row['distance_param'])}
    return country_to_group_map, group_params


def nearest_year_choice(df, id_col, value_col, target_year):
    """One row per id: the value from the year closest to target_year, later years winning
    ties, exactly as the source's fallback sorts. Returns id, year_used, year_distance and
    the value column."""
    rows = []
    for row_id, sub in df.groupby(id_col):
        sub = sub.assign(year_distance=(sub['year'] - target_year).abs())
        best = sub.sort_values(['year_distance', 'year'], ascending=[True, False]).iloc[0]
        rows.append({id_col: row_id, 'year_used': int(best['year']),
                     'year_distance': int(best['year_distance']), value_col: best[value_col]})
    return pd.DataFrame(rows)


def air_travel_value_by_country(arrivals, crosswalk, airfare, target_year):
    """Arrivals by origin region, priced at the staged region-to-region 2019 airfare matrix.

    Per destination country the year is the nearest one with any nonzero region-level
    arrivals, so a country's origin-region shares all come from one year. Arrivals from or to
    the unpriced residual region stay counted but unpriced, which the coverage share exposes.

    Args:
        arrivals (pd.DataFrame): iso3_r250_id, origin_region, year, arrivals.
        crosswalk (pd.DataFrame): iso3_r250_id, unwto_region.
        airfare (pd.DataFrame): origin_region, destination_region, predicted_fare_2019_usd.
        target_year (int): the base year.

    Returns:
        pd.DataFrame: iso3_r250_id, total_arrivals, priced_arrivals, air_travel_value,
        arrivals_coverage_share.
    """
    totals = arrivals.groupby(['iso3_r250_id', 'year'])['arrivals'].sum().reset_index()
    totals = totals[totals['arrivals'] > 0]
    year_choice = nearest_year_choice(totals, 'iso3_r250_id', 'arrivals', target_year)[
        ['iso3_r250_id', 'year_used']]

    chosen = arrivals.merge(year_choice, left_on=['iso3_r250_id', 'year'],
                            right_on=['iso3_r250_id', 'year_used'])
    merged = chosen.merge(
        crosswalk.rename(columns={'unwto_region': 'destination_region'}),
        on='iso3_r250_id', how='left')
    merged = merged.merge(airfare, on=['origin_region', 'destination_region'], how='left')
    merged['priced_arrivals'] = np.where(
        merged['predicted_fare_2019_usd'].notna(), merged['arrivals'], 0.0)
    merged['value'] = merged['arrivals'] * merged['predicted_fare_2019_usd']

    out = merged.groupby('iso3_r250_id').agg(
        total_arrivals=('arrivals', 'sum'),
        priced_arrivals=('priced_arrivals', 'sum'),
        air_travel_value=('value', lambda s: s.sum(skipna=True))).reset_index()
    out['arrivals_coverage_share'] = np.where(
        out['total_arrivals'] > 0, out['priced_arrivals'] / out['total_arrivals'], np.nan)
    return out


UNWTO_ARRIVAL_REGIONS = ('Africa', 'Americas', 'East Asia and the Pacific', 'Europe',
                         'Middle East', 'South Asia', 'Other not classified')


def clean_unwto_arrivals_by_region(df):
    """The Inbound Tourism-Regions sheet (read from its own header row) as a tidy arrivals
    table: iso3_r250_id, origin_region, year, arrivals (the sheet reports thousands)."""
    region_col = 'Unnamed: 6'
    df = df[df[region_col].isin(UNWTO_ARRIVAL_REGIONS)].copy()
    df['C.'] = df['C.'].ffill()
    year_cols = [c for c in df.columns if isinstance(c, int)]
    tidy = df.melt(id_vars=['C.', region_col], value_vars=year_cols,
                   var_name='year', value_name='arrivals_thousands')
    tidy = tidy.rename(columns={'C.': 'iso3_r250_id', region_col: 'origin_region'})
    tidy['origin_region'] = tidy['origin_region'].replace({'Other not classified': 'Other'})
    tidy['iso3_r250_id'] = tidy['iso3_r250_id'].astype(int)
    tidy['arrivals'] = pd.to_numeric(tidy['arrivals_thousands'], errors='coerce') * 1000.0
    return tidy[['iso3_r250_id', 'origin_region', 'year', 'arrivals']].dropna(subset=['arrivals'])


def allocate_overnights_array(hotel_array, country_array, country_overnights_map):
    """National overnight totals spread proportionally over each country's hotel pixels."""
    result = np.zeros_like(hotel_array, dtype=np.float32)
    for country_id, total_overnights in country_overnights_map.items():
        country_mask = (country_array == country_id)
        hotels_in_country = np.sum(hotel_array[country_mask])
        if hotels_in_country > 0:
            result[country_mask] = (hotel_array[country_mask] / hotels_in_country) * total_overnights
    return result


# --- Overnights (UNWTO panel + hotel allocation) ---
# The workbook keeps domestic and inbound accommodation on their own sheets, and the panel is
# the two stacked.
UNWTO_ACCOMMODATION_SHEETS = {'domestic': 'Domestic Tourism-Accommodation',
                              'international': 'Inbound Tourism-Accommodation'}
# Each sheet opens with banner rows above its real header, so the table is located by its first
# country row rather than by a fixed offset.
UNWTO_FIRST_COUNTRY = 'AFGHANISTAN'
UNWTO_COUNTRY_COLUMN_INDEX = 3


def unwto_header_row_index(banner_sheet):
    """Rows to skip so that a re-read starts at the sheet's own header.

    Args:
        banner_sheet (pd.DataFrame): the sheet read straight through, so its banner row is
            standing in as the header and the real header is its first row.

    Returns:
        int: the position of the first country row, which is the header row's position once the
        banner is no longer consuming a row.
    """
    country_column = banner_sheet.iloc[:, UNWTO_COUNTRY_COLUMN_INDEX]
    return banner_sheet[country_column == UNWTO_FIRST_COUNTRY].index[0]


def extract_clean_overnights(df, tourism_type):
    """One UNWTO accommodation sheet -> tidy (iso3_r250_id, name, type, year, overnights)."""
    df = df.copy()
    df.columns = df.columns.map(str)
    iso3_r250_id_col = 'C.'
    country_col = 'Basic data and indicators'
    indicator_col = 'Unnamed: 6'
    type_col = 'Unnamed: 5'
    for col in (iso3_r250_id_col, country_col, indicator_col, type_col):
        df[col] = df[col].ffill()

    overnight_df = df[df[indicator_col].str.contains('Overnights', na=False)]
    year_cols = [col for col in df.columns if col.isdigit()]
    overnight_df = overnight_df[[iso3_r250_id_col, country_col, type_col] + year_cols]

    tidy = overnight_df.melt(id_vars=[iso3_r250_id_col, country_col, type_col],
                             var_name='year', value_name='overnights')
    tidy = tidy.rename(columns={iso3_r250_id_col: 'iso3_r250_id', country_col: 'unwto_name',
                                type_col: 'overnight_type'})
    tidy['tourism_type'] = tourism_type
    tidy['year'] = tidy['year'].astype(int)
    tidy['overnights'] = pd.to_numeric(tidy['overnights'], errors='coerce')
    return tidy


def clean_unwto_data(sheets_by_tourism_type):
    """UNWTO accommodation sheets -> a wide overnight panel per country-year.

    The sheets are tidied and pivoted; per tourism type the combined overnights column prefers
    the Total row and falls back to Hotels-and-similar where Total is missing.

    Args:
        sheets_by_tourism_type (dict): tourism type -> that type's accommodation sheet, read
            from its own header row down (the task module's read_unwto_sheets).

    Returns:
        pd.DataFrame: iso3_r250_id, unwto_name, year and the per-type overnight columns.
    """
    rows_by_type = {tourism_type: extract_clean_overnights(sheet, tourism_type)
                    for tourism_type, sheet in sheets_by_tourism_type.items()}
    panel_df = pd.concat(list(rows_by_type.values()), ignore_index=True)
    panel_df['overnight_type'] = panel_df['overnight_type'].str.lower().str.strip()

    pivoted = panel_df.pivot_table(index=['iso3_r250_id', 'unwto_name', 'year'],
                                   columns=['overnight_type', 'tourism_type'],
                                   values='overnights', aggfunc='first').reset_index()
    pivoted.columns = ['iso3_r250_id', 'unwto_name', 'year'] + [
        f'{overnight_type}_overnights_{tourism_type}'
        for overnight_type, tourism_type in pivoted.columns[3:]]

    for tourism_type in ('domestic', 'international'):
        total_col = f'total_overnights_{tourism_type}'
        hotel_col = f'hotels and similar establishments_overnights_{tourism_type}'
        pivoted[f'overnights_{tourism_type}'] = np.where(
            pivoted[total_col].notna(), pivoted[total_col], pivoted[hotel_col])

    pivoted = pivoted.rename(columns={
        'hotels and similar establishments_overnights_domestic': 'hotel_overnights_domestic',
        'hotels and similar establishments_overnights_international': 'hotel_overnights_international'})
    final_cols = ['iso3_r250_id', 'unwto_name', 'year']
    for tourism_type in ('domestic', 'international'):
        final_cols += [f'total_overnights_{tourism_type}', f'hotel_overnights_{tourism_type}',
                       f'overnights_{tourism_type}']
    final_df = pivoted[[col for col in final_cols if col in pivoted.columns]]
    return final_df


def build_country_overnights_map(overnight_df, target_year):
    """Panel -> ({iso3_r250_id: overnights}, substitution rows), nearest year standing in.

    UNWTO reporting is patchy, so a country with no row at the target year takes its nearest
    year with positive overnights (later years win ties). The substitution rows say exactly
    which year each country used and how far off it was; a country with no usable year in ANY
    year is absent from both returns and reads downstream as missing, never as zero.
    """
    df = overnight_df.copy()
    df['total_overnights'] = df[['overnights_domestic', 'overnights_international']].sum(
        axis=1, min_count=1)
    df = df[df['total_overnights'] > 0]
    if df.empty:
        return {}, []
    choice = nearest_year_choice(df[['iso3_r250_id', 'year', 'total_overnights']],
                                 'iso3_r250_id', 'total_overnights', target_year)
    names = df.drop_duplicates('iso3_r250_id').set_index('iso3_r250_id')['unwto_name']
    country_overnights_map = dict(zip(choice['iso3_r250_id'], choice['total_overnights']))
    substitution_rows = [{
        'iso3_r250_id': row['iso3_r250_id'], 'unwto_name': names.get(row['iso3_r250_id'], ''),
        'year_used': row['year_used'], 'target_year': target_year,
        'year_distance': row['year_distance'], 'total_overnights': row['total_overnights'],
    } for _, row in choice.iterrows()]
    return country_overnights_map, substitution_rows


