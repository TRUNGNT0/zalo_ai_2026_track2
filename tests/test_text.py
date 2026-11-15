"""Test chuẩn hóa văn bản."""
from datamate.text import colnorm, norm


def test_norm_and_colnorm():
    assert norm("  Đơn   HÀNG Hủy ") == "don hang huy"
    assert colnorm("Số lượng (SL)") == "so_luong_sl"
