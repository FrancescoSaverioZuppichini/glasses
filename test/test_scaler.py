from glasses.models.utils.scaler import CompoundScaler


def test_scaler():
    scaler = CompoundScaler()
    widths = [32, 64]
    depths = [2,3]
    widths_s, depths_s = scaler(1, 1, widths=widths, depths=depths)

    assert widths_s == widths
    assert depths_s == depths_s

    widths_s, depths_s = scaler(1.1, 1.2, widths=widths, depths=depths)

    for w, w_s in zip(widths, widths_s):
        assert w <= w_s
    
    for d, d_s in zip(depths, depths_s):
        assert d <= d_s
