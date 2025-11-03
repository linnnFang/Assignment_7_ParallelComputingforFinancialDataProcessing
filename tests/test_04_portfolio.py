def test_portfolio_aggregation(portfolio_path):
    import json
    from portfolio import aggregate_portfolio_sequential, aggregate_portfolio_parallel

    with open(portfolio_path, 'r', encoding='utf-8') as f:
        tree = json.load(f)

    seq = aggregate_portfolio_sequential(tree)
    par = aggregate_portfolio_parallel(tree)

    # Required keys
    for obj in (seq, par):
        assert 'name' in obj and 'total_value' in obj and 'aggregate_volatility' in obj and 'max_drawdown' in obj
        assert 'positions' in obj

    # Totals must be close (deterministic RNG based on symbol)
    assert abs(seq['total_value'] - par['total_value']) < 1e-9
    assert abs(seq['aggregate_volatility'] - par['aggregate_volatility']) < 1e-9
    assert abs(seq['max_drawdown'] - par['max_drawdown']) < 1e-12
