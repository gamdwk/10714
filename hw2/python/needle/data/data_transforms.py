import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        #  H 代表图像的高度，W 代表图像的宽度，C 代表图像的通道数
        """
        H, W, C = img.shape
            for i in range(H):
                for j in range(W // 2 + 1):
                    swap img[i][j] img[i][W - j - 1]
        """
        if flip_img:
            return np.fliplr(img)
        else:
            return img
            
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        
        padded_image = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        x0 = shift_x + self.padding
        x1 = x0 + img.shape[0]
        y0 = shift_y + + self.padding
        y1 = y0 + img.shape[1]
        return padded_image[x0:x1, y0:y1]
        ### END YOUR SOLUTION
