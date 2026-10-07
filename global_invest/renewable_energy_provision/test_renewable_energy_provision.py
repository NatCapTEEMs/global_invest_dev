"""Unit tests for renewable_energy_provision's valuation.

Every pipeline step is pinned on a hand-built table small enough that the expected number is
written out in the assertion: the IRENA aggregation, the price conversion and join, the valued
concatenation, the base-year selection and the per-resource split.
"""
from types import SimpleNamespace

import pandas as pd

from global_invest.renewable_energy_provision import renewable_energy_provision_functions as rf


def _irena_frame():
    """Wind split over two sub-technology rows in one country-year, one solar row, one fossil
    row that the valued-resource selection must drop."""
    return pd.DataFrame({
        'Year': [2019, 2019, 2019, 2019],
        'ISO3 code': ['AAA', 'AAA', 'AAA', 'AAA'],
        'Country': ['Aaaland'] * 4,
        'Group Technology': ['Wind energy', 'Wind energy', 'Solar energy', 'Fossil fuels'],
        'Electricity Generation (GWh)': [30.0, 70.0, 10.0, 500.0],
    })


def test_generation_by_technology_sums_subtechnologies_and_drops_fossil():
    frames = rf.generation_by_technology(_irena_frame())
    assert len(frames) == len(rf.SUBSERVICE_TECHNOLOGIES)
    wind, solar, geothermal = frames
    assert wind['Electricity Generation (GWh)'].tolist() == [100.0]   # 30 + 70
    assert solar['Electricity Generation (GWh)'].tolist() == [10.0]
    assert len(geothermal) == 0                                       # no geothermal rows at all


def test_price_in_usd_per_gwh_converts_and_renames_for_the_join():
    priced = rf.price_in_usd_per_gwh(pd.DataFrame({
        'Economy ISO3': ['AAA'], 'Economy Name': ['Aaaland'], 'Year': [2019], 'Price': [8.0]}))
    assert priced['Price (USD/GWh)'].tolist() == [80000.0]
    assert 'ISO3 code' in priced.columns and 'Country' in priced.columns


def test_merge_price_onto_generation_is_inner_and_keeps_one_country_column():
    price = rf.price_in_usd_per_gwh(pd.DataFrame({
        'Economy ISO3': ['AAA'], 'Economy Name': ['Aaaland'], 'Year': [2019], 'Price': [8.0]}))
    generation = pd.DataFrame({
        'Year': [2019, 2019], 'ISO3 code': ['AAA', 'BBB'], 'Country': ['Aaaland', 'Bbbland'],
        'Group Technology': ['Wind energy'] * 2, 'Electricity Generation (GWh)': [100.0, 40.0]})

    merged = rf.merge_price_onto_generation(price, [generation])[0]

    assert merged['ISO3 code'].tolist() == ['AAA']    # BBB has no price row
    assert [c for c in merged.columns if c.startswith('Country')] == ['Country']


def test_valued_generation_prices_generation_times_natures_share():
    priced = pd.DataFrame({
        'ISO3 code': ['AAA'], 'Country': ['Aaaland'], 'Year': [2019],
        'Group Technology': ['Wind energy'],
        'Price (USD/GWh)': [80000.0], 'Electricity Generation (GWh)': [100.0]})
    attribution = pd.DataFrame({'Country': ['Aaaland'], 'Year': [2019], 'nat_contrib': [0.25]})

    valued = rf.valued_generation([priced], attribution)

    assert valued['renewable_energy_provision_gep'].tolist() == [2000000.0]  # 0.25 x 80000 x 100


def test_base_year_valued_rows_keep_the_base_year_and_positive_values_only():
    valued = pd.DataFrame({
        'ISO3 code': ['AAA', 'AAA', 'BBB'], 'Country': ['Aaaland', 'Aaaland', 'Bbbland'],
        'Year': [2019, 2018, 2019], 'Group Technology': ['Wind energy'] * 3,
        'Price (USD/GWh)': [80000.0] * 3, 'Electricity Generation (GWh)': [100.0] * 3,
        'nat_contrib': [0.25, 0.25, -0.1],
        'renewable_energy_provision_gep': [2000000.0, 2000000.0, -800000.0]})

    out = rf.base_year_valued_rows(valued, 2019)

    # 2018 is not the base year, and BBB's negative value comes out of the rent attribution.
    assert out['iso3_r250_label'].tolist() == ['AAA']
    assert 'ISO3 code' not in out.columns


def test_split_by_resource_keys_each_frame_by_its_technology_label():
    df = pd.DataFrame({
        'iso3_r250_label': ['AAA', 'AAA'], 'Country': ['Aaaland', 'Aaaland'], 'Year': [2019] * 2,
        'Group Technology': ['Wind energy', 'Solar energy'],
        'Price (USD/GWh)': [80000.0] * 2, 'Electricity Generation (GWh)': [100.0, 10.0],
        'nat_contrib': [0.25] * 2, 'renewable_energy_provision_gep': [2000000.0, 200000.0]})

    by_resource = rf.split_by_resource(df)

    assert set(by_resource) == {'Wind energy', 'Solar energy'}
    wind = by_resource['Wind energy']
    assert wind['energy_prod_GWh'].tolist() == [100.0]
    assert wind['Country_Name'].tolist() == ['Aaaland']


def test_renewable_energy_gep_is_generation_priced_times_natures_share():
    """100 GWh at 50 USD/GWh with a 0.2 nature share is 1,000 USD."""
    assert rf.renewable_energy_gep(0.2, 50.0, 100.0) == 1000.0
    assert rf.renewable_energy_gep(0.0, 50.0, 100.0) == 0.0


def test_the_price_conversion_turns_cents_per_kwh_into_usd_per_gwh():
    """One million kilowatt-hours per gigawatt-hour, over a hundred cents per dollar."""
    assert rf.CENTS_PER_KWH_TO_USD_PER_GWH == 1e6 / 100
    assert 8.0 * rf.CENTS_PER_KWH_TO_USD_PER_GWH == 80000.0


def test_es_config_row_hydrates_renewable_energy(tmp_path):
    from global_invest import utilities
    p = SimpleNamespace()
    p.input_dir = str(tmp_path / 'input')
    p.get_path = lambda *a, **k: '/resolved/' + '/'.join(a)
    utilities.hydrate_es_config(p, 'renewable_energy_provision', log=lambda *a: None)
    assert p.gep_base_year == 2019


# ---------------------------------------------------------------------------
# The regional fill: a country with no rent row takes its region's mean share.
# ---------------------------------------------------------------------------

def test_a_country_without_a_rent_row_takes_its_subregions_mean_share():
    """The Netherlands shape: a real generator with no attribution row must not vanish.

    Two neighbours carry shares 0.4 and 0.6, so the missing country fills at 0.5 and its
    generation is valued rather than dropped. A country WITH a row keeps its own share.
    """
    import pandas as pd
    from global_invest.renewable_energy_provision import renewable_energy_provision_functions as rf
    priced = [pd.DataFrame({'Country': ['A', 'B', 'C'], 'Year': [2019] * 3,
                            'ISO3 code': ['AAA', 'BBB', 'CCC'],
                            'Group Technology': ['Wind energy'] * 3,
                            'Price (USD/GWh)': [100_000.0] * 3,
                            'Electricity Generation (GWh)': [10.0, 10.0, 10.0]})]
    att = pd.DataFrame({'Country': ['A', 'B'], 'Year': [2019, 2019], 'nat_contrib': [0.4, 0.6]})
    regions = pd.DataFrame({'Country': ['A', 'B', 'C'], 'Sub-region': ['R1'] * 3, 'Region': ['W'] * 3})
    out = rf.valued_generation(priced, att, country_regions=regions).set_index('Country')
    assert out.loc['A', 'renewable_energy_provision_gep'] == 400_000.0
    assert out.loc['C', 'renewable_energy_provision_gep'] == 500_000.0     # filled at 0.5
    assert out.loc['C', 'attribution_source'] == 'regional mean'
    assert out.loc['A', 'attribution_source'] == 'country'


def test_a_measured_negative_share_is_kept_not_filled():
    """Negative rent is an answer, not a gap: the fill must not overwrite it."""
    import pandas as pd
    from global_invest.renewable_energy_provision import renewable_energy_provision_functions as rf
    priced = [pd.DataFrame({'Country': ['A', 'B'], 'Year': [2019] * 2,
                            'ISO3 code': ['AAA', 'BBB'],
                            'Group Technology': ['Wind energy'] * 2,
                            'Price (USD/GWh)': [100_000.0] * 2,
                            'Electricity Generation (GWh)': [10.0, 10.0]})]
    att = pd.DataFrame({'Country': ['A', 'B'], 'Year': [2019, 2019], 'nat_contrib': [-0.05, 0.6]})
    regions = pd.DataFrame({'Country': ['A', 'B'], 'Sub-region': ['R1'] * 2, 'Region': ['W'] * 2})
    out = rf.valued_generation(priced, att, country_regions=regions).set_index('Country')
    assert out.loc['A', 'nat_contrib'] == -0.05
