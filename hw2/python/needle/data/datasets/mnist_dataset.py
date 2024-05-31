from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip, struct
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms=transforms)
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        s = self.images[0].size
        self.images = self.images.reshape((self.images.shape[0], 28, 28, s // (28*28)) )
        self.num_classes = np.max(self.labels) + 1
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.images[index]
        if len(imgs.shape) == 4:
            for i in range(imgs.shape[0]):
                imgs[i] = self.apply_transforms(imgs[i])
        else:
            imgs = self.apply_transforms(imgs)
        return (imgs.reshape(imgs.shape[0], imgs[0].size), self.labels[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION
        
        
        
def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, 'rb') as image_file:
        images_data = image_file.read()
        meta_data = struct.unpack(">iii", images_data[4:16])
        # print(meta_data)
        image_num = meta_data[0]
        row = meta_data[1]
        line = meta_data[2]
        X = np.frombuffer(images_data[16:], dtype=np.uint8)
        X = X.astype(np.float32)
        # normalize
        X = (X - X.min()) / (X.max() - X.min())
        X = np.reshape(X, (image_num, row * line))

    with gzip.open(label_filename, 'rb') as label_file:
        labels_data = label_file.read()
        labels_num = struct.unpack(">i", labels_data[4:8])[0]
        assert labels_num == image_num
        y = np.frombuffer(labels_data[8:], dtype=np.uint8)
    return X, y
    ### END YOUR SOLUTION