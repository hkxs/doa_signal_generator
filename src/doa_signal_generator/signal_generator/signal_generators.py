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


def gen_complex_tone(frequency: float, num_samples: int, sample_rate: float) -> npt.NDArray:
    """
    Generate a complex tone signal

    Parameters
    ----------
    frequency: float
        The frequency of the tone
    num_samples: int
        Number of samples to generate
    sample_rate:
        The sampling frequency

    Returns
    -------
    npt.NDArray
        The generated complex signal
    """
    t = np.arange(num_samples) / sample_rate
    return np.exp(2j * np.pi * frequency * t)


def gen_sine_tone(frequency: float, num_samples: int, sample_rate: float) -> npt.NDArray:
    """
    Generate a sine wave with a defined frequency

    Parameters
    ----------
    frequency: float
        The frequency of the tone
    num_samples: int
        Number of samples to generate
    sample_rate:
        The sampling frequency

    Returns
    -------
    npt.NDArray
        The generated complex signal
    """
    samples = np.arange(num_samples) / sample_rate
    return np.sin(2 * np.pi * frequency * samples)
