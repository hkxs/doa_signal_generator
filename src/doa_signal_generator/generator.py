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

import matplotlib.pyplot as plt
import numpy.typing as npt
from doa_signal_generator.arrays import UniformLinealArray
from doa_signal_generator.signal_generator import gen_complex_tone


def apply_steering_vector(signal: npt.NDArray, steering_vector: npt.NDArray) -> npt.NDArray:
    """
    Apply a steering vector to a signal

    Parameters
    ----------
    signal: npt.NDArray
        Input signal to the array
    steering_vector
        Steering vector to be applied
    Returns
    -------
    npt.NDArray
        The output signal with steering vector applied
    """
    signal = signal.reshape(1, -1)
    steering_vector = steering_vector.reshape(1, -1).T
    return steering_vector @ signal


signal = gen_complex_tone(frequency=440, num_samples=500, sample_rate=44100)
ula = UniformLinealArray(4, 0.5)
a = ula.get_steering_vector(20)

output = apply_steering_vector(signal=signal, steering_vector=a)

for sensor in range(0, 4):
    plt.plot(output[sensor, :].real[:200], label=f'sensor {sensor}')
plt.show()
