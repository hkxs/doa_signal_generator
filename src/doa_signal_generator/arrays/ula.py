#  Copyright 2024 Hkxs
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the “Software”), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import numpy as np
import numpy.typing as npt

from doa_signal_generator.arrays.base_array import BaseArray


class UniformLinealArray(BaseArray):
    def __init__(self, num_elements, element_spacing) -> None:
        self.spacing = element_spacing
        super().__init__(num_elements)

    def get_steering_vector(self, angle_degrees: float) -> npt.NDArray:
        """
        Generate a steering vector for a uniform linear array

        Parameters
        ----------
        angle_degrees: float
            Angle of arrival in degrees

        Returns
        -------
        npt.NDArray
            The generating steering vector
        """
        theta = np.pi * angle_degrees / 180
        return np.exp(-2j * np.pi * self.spacing * np.arange(self.num_elements) * np.cos(theta))
