import polars as pl

def test_ingestion_equivalence(csv_path):
    from data_loader import load_pandas, load_polars, sanity_check_equivalence
    df_pd, prof_pd = load_pandas(csv_path)
    df_pl, prof_pl = load_polars(csv_path)

    # Basic schema checks
    assert set(['timestamp','symbol','price']).issubset(set(df_pd.columns))
    assert set(['timestamp','symbol','price']).issubset(set(df_pl.columns))

    # Ensure some rows
    assert len(df_pd) > 0
    assert df_pl.height > 0

    # Sanity equivalence (row counts, sample head/tail matches)
    assert sanity_check_equivalence(df_pd, df_pl), "Pandas/Polars ingestion outputs differ"
