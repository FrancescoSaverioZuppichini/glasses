from glasses.nn.models import VisionModule

def test_VisionModule():
    m = VisionModule()
    assert m.device.type == 'cpu'
    m.summary()