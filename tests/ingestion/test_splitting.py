"""Test chia business fit/holdout."""
from datamate.ingestion.datasets.splitting import BusinessSplitter


def test_holdout_is_stable_and_order_independent():
    ids = [f"b{i}" for i in range(30)]
    a = BusinessSplitter.holdout_businesses(ids, 6)
    assert len(a) == 6 and len(set(a)) == 6
    assert BusinessSplitter.holdout_businesses(list(reversed(ids)) + ids, 6) == a


def test_holdout_smaller_than_total():
    assert set(BusinessSplitter.holdout_businesses(["x", "y"], 5)) == {"x", "y"}
