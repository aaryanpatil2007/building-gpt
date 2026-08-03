import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        currans = x
        for i in range(len(weights)):
            currans = currans @ weights[i] + biases[i]
            if i != len(weights) - 1:
                currans = np.maximum(0.0, currans)
      
        return np.round(currans, 5)
        