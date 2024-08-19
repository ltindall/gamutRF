import logging
import sys 
import numpy as np 

from scipy import signal 

try:

    from gnuradio import gr  # pytype: disable=import-error
    from cupyx.scipy import signal as cupy_signal

except (ModuleNotFoundError, ImportError) as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)

class dc_spike_removal(gr.sync_block):
    """
    docstring for block dc_spike_removal
    """
    def __init__(self, ratio=0.995):
        gr.sync_block.__init__(self,
            name="dc_spike_removal",
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self.ratio = ratio
        self.d_avg_real = 0
        self.d_avg_img = 0

    def work(self, input_items, output_items):
        input = input_items[0]
        output = output_items[0]

        for i in range(len(input)):
            self.d_avg_real = self.ratio * (input[i].real - self.d_avg_real) + self.d_avg_real
            self.d_avg_img = self.ratio * (input[i].imag - self.d_avg_img) + self.d_avg_img

            output[i] = np.complex64(complex(real=input[i].real - self.d_avg_real, imag=input[i].imag - self.d_avg_img))

        return len(input)


class dc_spike_detrend(gr.sync_block):
    """
    docstring for block dc_spike_detrend
    """
    def __init__(self, length=1024):
        gr.sync_block.__init__(self,
            name="dc_spike_detrend",
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self.length = length

    def work(self, input_items, output_items):
        input = input_items[0]
        n_output = len(input)

        # output_items[0][:] = input_items[0]
        # output_items[0][:] = input_items[0] - cupy.mean(input_items[0])
        # output_items[0][:] = signal.detrend(input, type="constant")
        output_items[0][:] = signal.detrend(input, type="linear", bp=np.arange(0, len(input), self.length))
        # output_items[0][:] = signal.detrend(input, type="linear")

        # self.total += n_output
        return n_output

