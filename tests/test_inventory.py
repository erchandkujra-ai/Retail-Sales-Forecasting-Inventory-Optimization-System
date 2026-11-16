from src.inventory import calculate_inventory_policy
from src.intermittent import croston_sba


def test_inventory_policy_non_negative():

    result = calculate_inventory_policy(
        history=[2, 3, 4, 5, 3, 6, 4],
        forecast_daily=4
    )

    assert result["safety_stock"] >= 0
    assert result["reorder_point"] >= 0
    assert result["target_stock"] >= result["reorder_point"]
    assert result["eoq_reference"] >= 0


def test_zero_demand_policy():

    result = calculate_inventory_policy(
        history=[0, 0, 0, 0, 0],
        forecast_daily=0
    )

    assert result["average_daily_demand"] == 0
    assert result["reorder_point"] == 0


def test_croston_sba():

    result = croston_sba(
        [0, 0, 3, 0, 0, 4, 0],
        horizon=5
    )

    assert len(result) == 5
    assert (result >= 0).all()


def test_croston_zero_demand():

    result = croston_sba(
        [0, 0, 0, 0],
        horizon=7
    )

    assert len(result) == 7
    assert (result == 0).all()