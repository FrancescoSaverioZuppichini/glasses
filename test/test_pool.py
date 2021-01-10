import torch
from glasses.nn.pool import SpatialPyramidPool

def test_spp():
    num_pools = [1,4,16]
    pool = SpatialPyramidPool(num_pools=num_pools)
    x = torch.randn((4, 256, 14, 14))
    out = pool(x)

    assert out.shape[0] == x.shape[0] 
    assert out.shape[1] == x.shape[1] 
    assert out.shape[2] == sum(num_pools)