import h5py
from generative_models import GenerativeBayes

def test_data_availability():
        f = h5py.File("sheet04/data/digits.h5")
        images = f["images"].value
        labels = f["labels"].value
        f.close()

        assert images.shape[1:] == (9,9) and images.shape[0] == labels.shape[0]
