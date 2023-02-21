import argparse
import matplotlib.pyplot as plt
import numpy as np 
import time

def round_half(number):
    """Round a number to the closest half integer.
    >>> round_off_rating(1.3)
    1.5
    >>> round_off_rating(2.6)
    2.5
    >>> round_off_rating(3.0)
    3.0
    >>> round_off_rating(4.1)
    4.0"""

    return round(number * 2) / 2


def follow(thefile):
    while True:
        line = thefile.readline()
        if not line or not line.endswith('\n'):
            time.sleep(0.1)
            continue
        yield line

def draw_waterfall(mesh, fig, data, cmap): 
    mesh.set_array(cmap(data))
    fig.colorbar(mesh)
    fig.canvas.draw()

    plt.pause(0.01)


def argument_parser():
    parser = argparse.ArgumentParser(
        description="waterfall plotter from scan data"
    )
    parser.add_argument(
        "--fftlog", default="fftlog.csv", type=str, help="base path for fft log file"
    )
    return parser          


def main():
    parser = argument_parser()
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(16,8))
    cmap = plt.get_cmap('turbo')
    db_min = -30
    db_max = 0
    waterfall_height = 100 # number of waterfall rows 
    scale = 1e6
    min_freq = 300e6/scale
    max_freq = 6e9/scale
    freq_resolution = 0.5e6/scale

    X,Y = np.meshgrid(np.linspace(min_freq,max_freq,int((max_freq-min_freq)/freq_resolution+1)),np.linspace(1,waterfall_height,waterfall_height))
    db_data = np.empty(X.shape)
    db_data.fill(np.nan)
    mesh = ax.pcolormesh(X,Y, db_data, shading="nearest")
    ax.set_xlabel("MHz")
    

    #waterfall = []
    waterfall_row = []
    freq_idx = 0
    db_idx = 2
    with open(args.fftlog,"r") as logfile: 
        loglines = follow(logfile)
        for line in loglines:
            # expected format = ['frequency', 'timestamp', 'dB']
            split_line = line.split()

            if len(split_line) != 3: 
                continue
            line_floats = [float(x) for x in split_line]

            if not waterfall_row or waterfall_row[-1][freq_idx] < line_floats[freq_idx]: 
                waterfall_row.append(line_floats)
            else: 
                #waterfall.append(waterfall_row)
                freq = np.array([round_half(item[freq_idx]) for item in waterfall_row])
                db = np.array([item[db_idx] for item in waterfall_row])
                idx = (1/freq_resolution)*(freq - min_freq)
                idx = idx.astype(int)
                db_data = np.roll(db_data, -1, axis=0)
                db_data[-1,:] = np.nan
                db = (db - db_min) / (db_max - db_min)
                db_data[-1][idx] = db
                draw_waterfall(mesh, fig, db_data, cmap)
                waterfall_row = [line_floats]
                

if __name__ == "__main__":
    main()
        
        

