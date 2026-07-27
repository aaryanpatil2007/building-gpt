import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        initial = x@w + b
        forward = 1 / (1 + np.exp(-initial))
        dL_dYhat = forward - y_true
        d_yhat_dinitial = forward * (1-forward)
        dL_dinitial = dL_dYhat * d_yhat_dinitial
        dL_dW = x * dL_dinitial
        dL_db = dL_dinitial
        return np.round(dL_dW, 5), np.round(dL_db, 5)
        
