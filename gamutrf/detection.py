import cv2 
import argparse
import datetime
import zstandard
import os
import re
import time
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import json

from ultralytics import YOLO
from roboflow import Roboflow


SAMPLE_FILENAME_RE = re.compile(r"^.+_([0-9]+)_([0-9]+)Hz_([0-9]+)sps\.(s\d+|raw).*$")
FFT_FILENAME_RE = re.compile(
    r"^.+_([0-9]+)_([0-9]+)points_([0-9]+)Hz_([0-9]+)sps\.(s\d+|raw).*$"
)
SAMPLE_DTYPES = {
    "s8": ("<i1", "signed-integer"),
    "s16": ("<i2", "signed-integer"),
    "s32": ("<i4", "signed-integer"),
    "u8": ("<u1", "unsigned-integer"),
    "u16": ("<u2", "unsigned-integer"),
    "u32": ("<u4", "unsigned-integer"),
    "raw": ("<f4", "float"),
}

def get_reader(filename):
    # nosemgrep:github.workflows.config.useless-inner-function
    def gzip_reader(x):
        return gzip.open(x, "rb")

    # nosemgrep:github.workflows.config.useless-inner-function
    def zst_reader(x):
        return zstandard.ZstdDecompressor().stream_reader(
            open(x, "rb"), read_across_frames=True
        )

    def default_reader(x):
        return open(x, "rb")

    if filename.endswith(".gz"):
        return gzip_reader
    if filename.endswith(".zst"):
        return zst_reader

    return default_reader

def is_fft(filename):
    return os.path.basename(filename).startswith("fft_")

def parse_filename(filename):
    # FFT is always float not matter the original sample type.
    if is_fft(filename):
        sample_type = "raw"
        match = FFT_FILENAME_RE.match(filename)
        try:
            timestamp = int(match.group(1))
            nfft = int(match.group(2))
            freq_center = int(match.group(3))
            sample_rate = int(match.group(4))
            # sample_type = match.group(3)
        except AttributeError:
            timestamp = None
            nfft = None
            freq_center = None
            sample_rate = None
            sample_type = None
    else:
        match = SAMPLE_FILENAME_RE.match(filename)
        nfft = None
        try:
            timestamp = int(match.group(1))
            freq_center = int(match.group(2))
            sample_rate = int(match.group(3))
            sample_type = match.group(4)
        except AttributeError:
            timestamp = None
            freq_center = None
            sample_rate = None
            sample_type = None

    sample_dtype, sample_type = SAMPLE_DTYPES.get(sample_type, (None, None))
    sample_bits = None
    sample_len = None
    if sample_dtype:
        if is_fft(filename):
            sample_dtype = np.float32
            sample_bits = 32
            sample_len = 4
        else:
            sample_dtype = np.dtype([("i", sample_dtype), ("q", sample_dtype)])
            sample_bits = sample_dtype[0].itemsize * 8
            sample_len = sample_dtype[0].itemsize * 2
    file_info = {
        "filename": filename,
        "freq_center": freq_center,
        "sample_rate": sample_rate,
        "sample_dtype": sample_dtype,
        "sample_len": sample_len,
        "sample_type": sample_type,
        "sample_bits": sample_bits,
        "nfft": nfft,
        "timestamp": timestamp,
    }
    return file_info


def read_samples(filename, sample_dtype, sample_bytes, seek_bytes=0, nfft=None, n=None):
    reader = get_reader(filename)


    with reader(filename) as infile:
        infile.seek(int(seek_bytes))

        sample_buffer = infile.read(n * nfft * sample_bytes)
        buffered_samples = int(len(sample_buffer) / sample_bytes)

        if buffered_samples == 0:
            print(filename)
            return None
        if buffered_samples / nfft != n:
            print("incomplete")
            # return None

        x1d = np.frombuffer(sample_buffer, dtype=sample_dtype, count=buffered_samples)
        return x1d["i"] + np.csingle(1j) * x1d["q"]

class DtypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.dtype):
            return obj.descr
        return json.JSONEncoder.default(self, obj)

def argument_parser():
    parser = argparse.ArgumentParser(description="Waterfall plotter from scan data")
    parser.add_argument(
        "sample_dir", type=str, help="Directory with sample zst files."
    )
    parser.add_argument(
        "--min_freq", type=float, help="Minimum frequency for plotting."
    )
    parser.add_argument(
        "--max_freq", type=float, help="Maximum frequency for plotting."
    )
    parser.add_argument(
        "--sampling_rate", default=100e6, type=float, help="Sampling rate."
    )
    parser.add_argument("--nfft", default=256, type=int, help="FFT length.")
    parser.add_argument(
        "--write_samples",
        default=2048,
        type=int,
        help="Number of samples written during scan.",
    )
    parser.add_argument(
        "--save_data",
        action="store_true", 
        help="Save processed image data that is sent to model."
    )

    return parser


def main():
    # ARG PARSE PARAMETERS
    parser = argument_parser()
    args = parser.parse_args()
    sample_dir = args.sample_dir
    min_freq = args.min_freq
    max_freq = args.max_freq
    nfft = args.nfft
    n = args.write_samples
    sps = args.sampling_rate
    save_data = args.save_data

    noverlap = 0#nfft // 8
    db_min = -220
    db_max = -60

    model = YOLO("/home/ltindall/ultralytics/runs/detect/yolov8s_exp_v05/weights/best.pt")

    cmap = plt.get_cmap("turbo")

    plot_min_freq = min_freq  # freq - (sps/2)
    plot_max_freq = max_freq  # freq + (sps/2)

    freq_resolution = sps / nfft
    max_idx = round((max_freq - min_freq) / freq_resolution)
    total_time = (nfft * n) / sps
    expected_time_bins = int((nfft * n) / (nfft - noverlap))
    X, Y = np.meshgrid(
        np.linspace(
            plot_min_freq,
            plot_max_freq,
            int((max_freq - min_freq) / freq_resolution + 1),
        ),
        np.linspace(0, total_time, expected_time_bins),
    )
    freq_bin_vals = X[0]
    spec_data = np.empty(X.shape)
    spec_data.fill(np.nan)

    ii = 0 
    processed_files = []
    while True:
        sample_files = [
            f
            for f in sorted(os.listdir(sample_dir))
            if f.startswith("sample") and f.endswith(".zst")
        ]

        needs_processing = []
        processing_batch = []
        for f in sample_files:
            f = os.path.join(sample_dir, f)
            file_info = parse_filename(f)
            freq_center = file_info["freq_center"]
            sample_rate = file_info["sample_rate"]
            if (
                (
                    ((freq_center + (sample_rate / 2)) >= min_freq)
                    and ((freq_center + (sample_rate / 2)) <= max_freq)
                )
                or (
                    ((freq_center - (sample_rate / 2)) >= min_freq)
                    and ((freq_center - (sample_rate / 2)) <= max_freq)
                )
            ) and f not in processed_files:
                if (
                    not processing_batch
                    or processing_batch[-1]["freq_center"] < freq_center
                ):
                    processing_batch.append(file_info)
                else:
                    needs_processing.append(processing_batch)
                    processing_batch = [file_info]

        # process files
        if needs_processing:

            spec_data.fill(np.nan)
            meta_data = {"sample_files":[]}
            for file_info in needs_processing[0]:
                filename = file_info["filename"]
                meta_data["sample_files"].append(file_info)
                sample_dtype = file_info["sample_dtype"]
                sample_bytes = file_info["sample_len"]
                freq_center = file_info["freq_center"]
                sample_rate = file_info["sample_rate"]
                sample_type = file_info["sample_type"]
                sample_bits = file_info["sample_bits"]
                timestamp = file_info["timestamp"]

                samples = read_samples(
                    filename, sample_dtype, sample_bytes, seek_bytes=0, nfft=nfft, n=n
                )
                if samples is None:
                    continue
                freq_bins, t_bins, S = signal.spectrogram(
                    samples,
                    sample_rate,
                    window=signal.hann(int(nfft), sym=True),
                    nperseg=nfft,
                    noverlap=noverlap,
                    detrend='constant',
                    return_onesided=False,
                )
                freq_bins = np.fft.fftshift(freq_bins)
                # print(f"{freq_bins.shape=}{freq_center + freq_bins=}")

                idx = np.array(
                    [
                        round((item - min_freq) / freq_resolution)
                        for item in freq_bins + freq_center
                    ]
                ).astype(int)

                S = np.fft.fftshift(S, axes=0)
                S = S.T
                S = 10 * np.log10(S)

                S_norm = (S - db_min) / (db_max - db_min)
   
                spec_data[
                    : S_norm.shape[0],
                    idx[np.flatnonzero((idx >= 0) & (idx <= max_idx))],
                ] = S_norm[:, np.flatnonzero((idx >= 0) & (idx <= max_idx))]
                processed_files.append(filename)
            
            #res = cv2.resize(cmap(spec_data)[:,:,:3], dsize=(1640, 640), interpolation=cv2.INTER_CUBIC)[:,:,::-1]
            res = cmap(spec_data)[:,:,:3] # remove alpha dimension
            res = res[::-1,:,:] # flip vertically
            res *= 255
            res = res.astype(int)
            res = np.ascontiguousarray(res, dtype=np.uint8)

            if save_data: 
                im = Image.fromarray(res)
                image_dir = Path(f"{sample_dir}/png/")
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"{ii}.png"
                im.save(image_path)
                meta_data["img_file"] = str(image_path)

            meta_data["id"] = ii
            meta_dir = Path(f"{sample_dir}/metadata/")
            meta_dir.mkdir(parents=True, exist_ok=True)
            json_object = json.dumps(meta_data, indent=4, cls=DtypeEncoder)
            with open(meta_dir / f"{ii}.json", "w") as outfile:
                outfile.write(json_object)
    
            if ii > 0: # bug in yolov8, name parameters is broken in predict()
                model.predictor.save_dir = Path(f"{sample_dir}/predictions/{ii}")
            results = model.predict(source=res[:,:,::-1], conf=0.05, save=True, save_txt=True, save_conf=True, project=f"{sample_dir}/predictions/", name=f"{ii}", exist_ok=True)
            

            ii += 1
                  

        else:
            print("waiting")
            quit()


if __name__ == "__main__":
    main()
