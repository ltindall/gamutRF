import argparse
import datetime
import matplotlib
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import shlex
import signal
import subprocess
import sys
import time

from findpeaks import findpeaks
from matplotlib.widgets import RangeSlider, TextBox
from timeit import default_timer as timer

from gamutrf.zmqreceiver import ZmqReceiver

sleep_time = 0.5

def read_log(log_file):
    while True:
        line = log_file.readline()
        if not line or not line.endswith("\n"):
            print("WAITING FOR SCANNER...\n")
            time.sleep(sleep_time)
            continue
        yield line

def draw_waterfall(mesh, fig, ax, data, cmap, background):
    start = timer()
    mesh.set_array(cmap(data))
    end = timer()
    ax.draw_artist(mesh)

def draw_title(ax, title, title_text):
    title_text["Time"] = str(datetime.datetime.now())
    title.set_text(str(title_text))
    ax.draw_artist(title)

def argument_parser():
    parser = argparse.ArgumentParser(description="waterfall plotter from scan data")
    parser.add_argument(
        "--min_freq", 
        default=300e6, 
        type=float, 
        help="Minimum frequency for plot."
    )
    parser.add_argument(
        "--max_freq", 
        default=6e9, 
        type=float, 
        help="Maximum frequency for plot."
    )
    parser.add_argument(
        "--sampling_rate", 
        default=100e6, 
        type=float, 
        help="Sampling rate."
    )
    parser.add_argument(
        "--nfft", 
        default=256, 
        type=int, 
        help="FFT length."
    )
    parser.add_argument(
        "--n_detect", 
        default=80, 
        type=int, 
        help="Number of detected signals to plot."
    )
    parser.add_argument(
        "--plot_snr", 
        action="store_true", 
        help="Plot SNR rather than power."
    )
    return parser

def main():
    # ARG PARSE PARAMETERS
    parser = argument_parser()
    args = parser.parse_args()
    min_freq = args.min_freq  # 5.7e9 #300e6
    max_freq = args.max_freq  # 5.9e9#6e9
    plot_snr = args.plot_snr
    top_n = args.n_detect
    fft_len = args.nfft
    sampling_rate = args.sampling_rate

    # OTHER PARAMETERS
    cmap = plt.get_cmap("turbo")
    db_min = -220
    db_max = -150
    snr_min = 0
    snr_max = 50
    waterfall_height = 100  # number of waterfall rows
    scale = 1e6

    freq_resolution = (
        sampling_rate / fft_len
    )  
    draw_rate = 1
    y_label_skip = 3

    # SCALING
    min_freq /= scale
    max_freq /= scale
    freq_resolution /= scale

    # DATA
    X, Y = np.meshgrid(
        np.linspace(
            min_freq, max_freq, int((max_freq - min_freq) / freq_resolution + 1)
        ),
        np.linspace(1, waterfall_height, waterfall_height),
    )
    freq_bins = X[0]
    db_data = np.empty(X.shape)
    db_data.fill(np.nan)
    freq_data = np.empty(X.shape)
    freq_data.fill(np.nan)

    fig = plt.figure(figsize=(28, 10), dpi=100)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    ax_psd = fig.add_subplot(3, 1, 1)
    ax = fig.add_subplot(3, 1, (2, 3))

    psd_db_resolution = 90
    XX, YY = np.meshgrid(
        np.linspace(
            min_freq, max_freq, int((max_freq - min_freq) / (freq_resolution) + 1)
        ),
        np.linspace(db_min, db_max, psd_db_resolution),
    )

    psd_x_edges = XX[0]
    psd_y_edges = YY[:, 0]

    mesh_psd = ax_psd.pcolormesh(XX, YY, np.zeros(XX[:-1, :-1].shape), shading="flat")
    (max_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="red",
        marker=",",
        linestyle=":",
        markevery=10,
        label="max",
    )
    (min_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="pink",
        marker=",",
        linestyle=":",
        markevery=10,
        label="min",
    )
    (mean_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="cyan",
        marker="^",
        markersize=8,
        fillstyle="none",
        linestyle=":",
        markevery=20,
        label="mean",
    )
    (var_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="magenta",
        marker="s",
        markersize=8,
        fillstyle="none",
        linestyle=":",
        markevery=20,
        label="variance",
    )
    (current_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="red",
        marker="o",
        markersize=8,
        fillstyle="none",
        linestyle=":",
        markevery=10,
        label="current",
    )
    ax_psd.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax_psd.set_ylabel("dB")

    mesh = ax.pcolormesh(X, Y, db_data, shading="nearest")

    top_n_lns = []
    for _ in range(top_n):
        (ln,) = ax.plot(
            [X[0][0]] * len(Y[:, 0]), Y[:, 0], color="brown", linestyle=":", alpha=0
        )
        top_n_lns.append(ln)

    ax.set_xlabel("MHz")
    ax.set_ylabel("Time")

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=db_min, vmax=db_max)

    if plot_snr:
        sm.set_clim(vmin=snr_min, vmax=snr_max)
    cbar_ax = fig.add_axes([0.92, 0.10, 0.03, 0.5])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("dB", rotation=0)

    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, va="center", ha="center")
    psd_title = ax_psd.text(
        0.5, 1.05, "", transform=ax_psd.transAxes, va="center", ha="center"
    )
    title_text = {}
    y_ticks = []
    y_labels = []
    cbar_ax.yaxis.set_animated(True)
    ax.yaxis.set_animated(True)
    plt.show(block=False)
    plt.pause(0.1)
    background = fig.canvas.copy_from_bbox(fig.bbox)
    background_psd = fig.canvas.copy_from_bbox(ax_psd.bbox)

    ax.draw_artist(mesh)

    fig.canvas.blit(ax.bbox)

    for ln in top_n_lns:
        ln.set_alpha(0.75)

    counter = 0
    
    scan_fres_resolution = 1e4
    zmqr = ZmqReceiver(addr="127.0.0.1", port=8001, scan_fres=scan_fres_resolution)
    def sig_handler(_sig=None, _frame=None):
        zmqr.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

        
    while True: 
        start = timer()
        start_read = timer()
        scan_config, scan_df = zmqr.read_buff()

        if scan_df is not None: 

            scan_df = scan_df[(scan_df.freq > min_freq) & (scan_df.freq < max_freq)]
            if scan_df.empty: 
                print(f"Scan is outside specified frequency range ({min_freq} to {max_freq}).")
                continue
            
            end_read = timer()
            scan_time = end_read - start_read
            end = timer()
            start = timer()

            idx = round((scan_df.freq - min_freq) / freq_resolution).values.flatten().astype(int)
            
            freq_data = np.roll(freq_data, -1, axis=0)
            freq_data[-1, :] = np.nan
          
            freq_data[-1][idx] = round(scan_df.freq / freq_resolution).values.flatten() * freq_resolution

            db = scan_df.db.values.flatten()

            db_data = np.roll(db_data, -1, axis=0)
            db_data[-1, :] = np.nan
            db_data[-1][idx] = db

  
            data, xedge, yedge = np.histogram2d(
                freq_data[~np.isnan(freq_data)].flatten(),
                db_data[~np.isnan(db_data)].flatten(),
                density=False,
                bins=[psd_x_edges, psd_y_edges],
            ) 

            data /= np.max(data)  

            np.set_printoptions(threshold=sys.maxsize)
            fig.canvas.restore_region(background)

            mesh_psd.set_array(cmap(data.T))

            top_n_bins = freq_bins[
                np.argsort(np.nanvar(db_data - np.nanmin(db_data, axis=0), axis=0))[
                    ::-1
                ][:top_n]
            ]

            for i, ln in enumerate(top_n_lns):
                ln.set_xdata([top_n_bins[i]] * len(Y[:, 0]))

            min_psd_ln.set_ydata(np.nanmin(db_data, axis=0))
            max_psd_ln.set_ydata(np.nanmax(db_data, axis=0))
            mean_psd_ln.set_ydata(np.nanmean(db_data, axis=0))
            var_psd_ln.set_ydata(np.nanvar(db_data, axis=0))
            current_psd_ln.set_ydata(db_data[-1])
            ax_psd.draw_artist(mesh_psd)

            ax_psd.draw_artist(min_psd_ln)

            ax_psd.draw_artist(max_psd_ln)
            ax_psd.draw_artist(mean_psd_ln)
            ax_psd.draw_artist(var_psd_ln)
            ax_psd.draw_artist(current_psd_ln)


            fig.canvas.blit(ax.yaxis.axes.figure.bbox)
            fig.canvas.blit(ax_psd.bbox)

            row_time = datetime.datetime.fromtimestamp(
                scan_df.ts.iloc[-1]
            )

            if counter % y_label_skip == 0:
                y_labels.append(row_time)
            else:
                y_labels.append("")
            y_ticks.append(waterfall_height)
            for j in range(len(y_ticks) - 2, -1, -1):
                y_ticks[j] -= 1
                if y_ticks[j] < 1:
                    y_ticks.pop(j)
                    y_labels.pop(j)

            ax.set_yticks(y_ticks, labels=y_labels)
            end = timer()

            counter += 1
            if counter % draw_rate == 0:

                draw_rate = 1
                start = timer()

                db_min = np.nanmin(db_data)
                db_max = np.nanmax(db_data)

                db_norm = (db_data - db_min) / (db_max - db_min)
                if plot_snr:
                    db_norm = ((db_data - np.nanmin(db_data, axis=0)) - snr_min) / (
                        snr_max - snr_min
                    )
                draw_waterfall(mesh, fig, ax, db_norm, cmap, background)

                draw_title(ax_psd, psd_title, title_text)
                
                sm.set_clim(vmin=db_min, vmax=db_max)
                cbar.update_normal(sm)
                cbar.draw_all()
                cbar_ax.draw_artist(cbar_ax.yaxis)
                fig.canvas.blit(cbar_ax.yaxis.axes.figure.bbox)
                for ln in top_n_lns:
                    ax.draw_artist(ln)
                ax.draw_artist(ax.yaxis)
                fig.canvas.blit(ax.yaxis.axes.figure.bbox)
                fig.canvas.blit(ax.bbox)
                fig.canvas.blit(cbar_ax.bbox)
                fig.canvas.blit(fig.bbox)
                fig.canvas.flush_events()
                end = timer()
                print(f"Plotting {row_time}")

            print("\n")

        else: 
            print("Waiting for scanner (ZMQ)...")
            time.sleep(1)


if __name__ == "__main__":
    main()
