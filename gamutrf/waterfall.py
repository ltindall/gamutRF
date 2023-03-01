import argparse
import os
import matplotlib.pyplot as plt
import numpy as np 
import time
from timeit import default_timer as timer
import datetime
from findpeaks import findpeaks
import signal
import subprocess
import sys
import shlex

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
            print("WAITING\n\n")
            time.sleep(0.5)
            continue
        yield line

def draw_waterfall(mesh, fig, data, cmap): 
    start = timer() 
    mesh.set_array(cmap(data))
    end = timer() 
    print(f"Set mesh {end-start}")
    start = timer() 
    fig.canvas.draw()
    end = timer() 
    print(f"Draw {end-start}")
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

    fig, ax = plt.subplots(figsize=(24,10), dpi=100)
    cmap = plt.get_cmap('turbo')
    db_min = -40
    db_max = 15
    waterfall_height = 100 # number of waterfall rows 
    scale = 1e6
    min_freq = 300e6/scale
    max_freq = 6e9/scale
    freq_resolution = 0.5e6/scale
    draw_rate = 1
    y_label_skip = 3

    X,Y = np.meshgrid(np.linspace(min_freq,max_freq,int((max_freq-min_freq)/freq_resolution+1)),np.linspace(1,waterfall_height,waterfall_height))
    db_data = np.empty(X.shape)
    db_data.fill(np.nan)
    mesh = ax.pcolormesh(X,Y, db_data, shading="nearest")
    ax.set_xlabel("MHz")
    ax.set_ylabel("Time")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=db_min, vmax=db_max)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('dB', rotation=0)
    y_ticks = []
    y_labels = []
    

    #waterfall = []
    waterfall_row = []
    freq_idx = 0
    timestamp_idx = 1
    db_idx = 2
    if not os.path.exists(args.fftlog): 
        print(f"Waiting for {args.fftlog}...")
        while not os.path.exists(args.fftlog): 
            time.sleep(1)
        print(f"Found {args.fftlog}. Starting waterfall plot.")
    

    fp = findpeaks(method="topology", imsize=db_data.shape, scale=True, togray=True, denoise=None)

    with open(args.fftlog,"r") as logfile: 
        loglines = follow(logfile)
        counter = 0
        start_read = timer() 
        for line in loglines:
            start = timer() 
            # expected format = ['frequency', 'timestamp', 'dB']
            split_line = line.split()

            line_floats = [float(x) for x in split_line]

            if not waterfall_row or waterfall_row[-1][freq_idx] < line_floats[freq_idx]: 
                waterfall_row.append(line_floats)
            else: 
                end_read = timer() 
                print(f"Read waterfall row {end_read-start_read} ")
                
                end = timer()
                #print(f"Read waterfall row {end-start}")
                start = timer() 
                #waterfall.append(waterfall_row)
                freq = np.array([round_half(item[freq_idx]) for item in waterfall_row])
                db = np.array([item[db_idx] for item in waterfall_row])
                idx = (1/freq_resolution)*(freq - min_freq)
                idx = idx.astype(int)
                db_data = np.roll(db_data, -1, axis=0)
                db_data[-1,:] = np.nan
                db = (db - db_min) / (db_max - db_min)
                db_data[-1][idx] = db

                # fit_start = timer() 
                # results = fp.fit(db_data)
                # fit_end = timer() 
                # print(f"Fit {fit_end-fit_start}")

                row_time = datetime.datetime.fromtimestamp(float(waterfall_row[-1][timestamp_idx]))

                if counter%y_label_skip == 0: 
                    y_labels.append(row_time)
                else: 
                    y_labels.append('')
                y_ticks.append(waterfall_height)
                for j in range(len(y_ticks)-2, -1, -1):
                    y_ticks[j] -= 1
                    if y_ticks[j] < 1: 
                        y_ticks.pop(j)
                        y_labels.pop(j)


                ax.set_yticks(y_ticks, labels=y_labels)
                end = timer() 
                print(f"Process row {end-start}")
                
                counter += 1
                if counter % draw_rate == 0: 
                    if (end_read - start_read ) < 1: 
                        draw_rate = 4
                        print(f"Draw rate = {draw_rate}")
                    else: 
                        draw_rate = 1
                        #draw_rate = int(end_read-start_read) + 1
                        print(f"Draw rate = {draw_rate}")
                    start = timer() 
                    draw_waterfall(mesh, fig, db_data, cmap)
                    end = timer() 
                    print(f"Redraw {end-start}")
                waterfall_row = [line_floats]
                start_read = timer() 
                

if __name__ == "__main__":

    # processes = []
    # commands = ([
    #     "gamutrf-scan --sdr=SoapyAIRT --freq-start=300e6 --freq-end=6e9 --tune-step-fft 2048 --samp-rate=124.5184e6 --nfft 256", 
    #     "gamutrf-sigfinder --freq-start=300e6 --freq-end=6e9 --promport=9009 --fftgraph fft.png --port 9005 --nfftplots 1 --buff_path /tmp/ --fftlog fftlog.csv",
    # ])
    # for command in commands: 
    #     #pro = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 
    #     #pro = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True, preexec_fn=os.setsid) 
    #     pro = subprocess.Popen(shlex.split(command),stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,  preexec_fn=os.setsid)
    #     _,_ = pro.communicate() 
    #     processes.append(pro)


    # def sigint_handler(_signo, _stack_frame):
    #     for process in processes: 
    #         os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    #     # Raises SystemExit(0):
    #     sys.exit(0)

    # signal.signal(signal.SIGINT, sigint_handler)

    main()

    

    
        

