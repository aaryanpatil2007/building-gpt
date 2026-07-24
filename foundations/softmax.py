import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted_array = z - np.max(z, axis=-1, keepdims=True)
        top = np.exp(shifted_array)
        bottom = np.sum(np.exp(shifted_array))
        return np.round((top / bottom), 4)