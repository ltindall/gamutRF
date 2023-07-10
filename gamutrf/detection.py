import cv2 
import argparse
import datetime
import zstandard
import os
import re
import time
import gzip
import bz2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import json

from ultralytics import YOLO
#from roboflow import Roboflow


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
    def bz2_reader(x):
        return bz2.open(x, "rb")

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

    if filename.endswith(".bz2"): 
        return bz2_reader
    if filename.endswith(".gz"):
        return gzip_reader
    if filename.endswith(".zst"):
        return zst_reader

    return default_reader

def is_fft(filename):
    return os.path.basename(filename).startswith("fft_")

def supported_filetype(filename): 
    supported_filetypes = ([
        ".zst",
        ".gz",
        ".bz2",
        ".raw",
        ".s8",
        ".s16",
        ".s32",
        ".u8",
        ".u16",
        ".u32",
    ]) 
    if not any([filename.endswith(supported) for supported in supported_filetypes]): 
        return False
    return True
    
def parse_filename(filename):
    # supported_filetypes = ([
    #     ".zst",
    #     ".gz",
    #     ".bz2",
    #     ".raw",
    #     ".s8",
    #     ".s16",
    #     ".s32",
    #     ".u8",
    #     ".u16",
    #     ".u32",
    # ])
    # if not any([filename.endswith(supported) for supported in supported_filetypes]): 
    #     return None
    # FFT is always float not matter the original sample type.
    if is_fft(filename):
        print("Skipping FFT file "+filename)
        return None
        sample_type = "raw"
        match = FFT_FILENAME_RE.match(filename)
        try:
            timestamp = int(match.group(1))
            nfft = int(match.group(2))
            freq_center = int(match.group(3))
            sample_rate = int(match.group(4))
            # sample_type = match.group(3)
        except AttributeError:
            return None

    else:
        match = SAMPLE_FILENAME_RE.match(filename)
        nfft = None
        try:
            timestamp = int(match.group(1))
            freq_center = int(match.group(2))
            sample_rate = int(match.group(3))
            sample_type = match.group(4)
        except AttributeError:
            print("Error reading file "+filename)
            return None
 

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


def read_samples(filename, sample_dtype, sample_bytes, seek_bytes=0, nfft=None, fft_count=None):
    print("Reading "+filename)
    reader = get_reader(filename)

    with reader(filename) as infile:
        infile.seek(int(seek_bytes))

        if fft_count is not None: 
            sample_buffer = infile.read(fft_count * nfft * sample_bytes)
        else: 
            sample_buffer = infile.read()

        buffered_samples = int(len(sample_buffer) / sample_bytes)

        if buffered_samples == 0:
            print("Error! No samples read from "+filename)
            return None
        if fft_count is not None and buffered_samples / nfft != fft_count:
            print("Incomplete sample file. Could not load the expected number of samples.")

        x1d = np.frombuffer(sample_buffer, dtype=sample_dtype, count=buffered_samples)
        return x1d["i"] + np.csingle(1j) * x1d["q"]

class DtypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.dtype):
            return obj.descr
        return json.JSONEncoder.default(self, obj)

def prepare_custom_spectrogram(min_freq, max_freq, sample_rate, nfft, fft_count, noverlap):  
    freq_resolution = sample_rate / nfft
    max_idx = round((max_freq - min_freq) / freq_resolution)
    total_time = (nfft * fft_count) / sample_rate
    expected_time_bins = int((nfft * fft_count) / (nfft - noverlap))
    X, Y = np.meshgrid(
        np.linspace(
            min_freq,
            max_freq,
            int((max_freq - min_freq) / freq_resolution + 1),
        ),
        np.linspace(0, total_time, expected_time_bins),
    )
    spectrogram_array = np.empty(X.shape)
    spectrogram_array.fill(np.nan)
    
    return spectrogram_array, max_idx, freq_resolution

def argument_parser():
    parser = argparse.ArgumentParser(
        description="Process sample files and run inference.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sample_dir", 
        type=str, 
        help="Directory with sample files.",
    )
    parser.add_argument(
        "--min_freq", 
        type=float, 
        default=None, 
        help="Minimum frequency for plotting.",
    )
    parser.add_argument(
        "--max_freq", 
        type=float, 
        default=None, 
        help="Maximum frequency for plotting.",
    )
    parser.add_argument(
        "--nfft", 
        type=int,
        default=256,
        help="FFT length.",
    )
    parser.add_argument(
        "--fft_count",
        default=None,
        type=int,
        help="Number of FFT operations to perform, or expected to perform.",
    )
    parser.add_argument(
        "--save_data",
        action="store_true", 
        help="Save processed image data that is sent to model.",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true", 
        help="Don't perform inference.",
    )

    return parser


def main():
    # ARG PARSE PARAMETERS
    parser = argument_parser()
    args = parser.parse_args()
    sample_dir = args.sample_dir
    nfft = args.nfft
    fft_count = args.fft_count
    save_data = args.save_data
    skip_inference = args.skip_inference
    min_freq = args.min_freq
    max_freq = args.max_freq
    if (min_freq is None and max_freq is not None) or (min_freq is not None and max_freq is None): 
        print("Error! If min_freq or max_freq is defined then both must be defined. Exiting.")
        return
    if min_freq is not None and max_freq is not None: 
        custom_spectrogram = True
    else: 
        custom_spectrogram = False

    # db_min = -220
    # db_max = -60
    noverlap = 0 #nfft // 8
    model = YOLO("/home/ltindall/ultralytics/runs/detect/yolov8s_exp_v05/weights/best.pt")
    cmap = plt.get_cmap("turbo")

    spectrogram_id = 0 
    processed_files = []

    wait_count = 0 
    wait_time = 1
    wait_count_limit = 5
    
    while True:
        unprocessed_files = [
            basefilename
            for basefilename in sorted(os.listdir(sample_dir))
            if os.path.isfile(os.path.join(sample_dir, basefilename)) and basefilename not in processed_files
        ]

        # Process files
        if unprocessed_files:

            meta_data = {}

            for basefilename in unprocessed_files: 
                processed_files.append(basefilename)
                # Load samples 
                file_info = parse_filename(os.path.join(sample_dir, basefilename))
                if file_info is None: 
                    continue
                samples = read_samples(
                    file_info["filename"], 
                    file_info["sample_dtype"], 
                    file_info["sample_len"], 
                    seek_bytes=0, 
                    nfft=nfft, 
                    fft_count=fft_count,
                )
                if samples is None:
                    print("Continuing...")
                    continue

                # Convert samples into spectrogram
                freq_bins, t_bins, spectrogram = signal.spectrogram(
                    samples,
                    file_info["sample_rate"],
                    window=signal.windows.hann(int(nfft), sym=True),
                    nperseg=nfft,
                    noverlap=noverlap,
                    detrend='constant',
                    return_onesided=False,
                )
                # FFT shift 
                freq_bins = np.fft.fftshift(freq_bins)
                spectrogram = np.fft.fftshift(spectrogram, axes=0)
                # Transpose spectrogram
                spectrogram = spectrogram.T
                # dB scale spectrogram
                spectrogram = 10 * np.log10(spectrogram)
                # Normalize spectrogram
                spectrogram_normalized = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram)) #(spectrogram - db_min) / (db_max - db_min)
                
                spectrogram_data = spectrogram_normalized

                if custom_spectrogram: 
                    if fft_count is None: 
                        fft_count = len(t_bins)
                    spectrogram_data, max_idx, freq_resolution = prepare_custom_spectrogram(
                        min_freq, 
                        max_freq, 
                        file_info["sample_rate"], 
                        nfft, 
                        fft_count, 
                        noverlap
                    )
                    idx = np.array(
                        [
                            round((item - min_freq) / freq_resolution)
                            for item in freq_bins + file_info["freq_center"]
                        ]
                    ).astype(int)
                    spectrogram_data[
                        : spectrogram_normalized.shape[0],
                        idx[np.flatnonzero((idx >= 0) & (idx <= max_idx))],
                    ] = spectrogram_normalized[:, np.flatnonzero((idx >= 0) & (idx <= max_idx))]
            
                # Spectrogram color transforms 
                #spectrogram_color = cv2.resize(cmap(spectrogram_data)[:,:,:3], dsize=(1640, 640), interpolation=cv2.INTER_CUBIC)[:,:,::-1]
                spectrogram_color = cmap(spectrogram_data)[:,:,:3] # remove alpha dimension
                spectrogram_color = spectrogram_color[::-1,:,:] # flip vertically
                spectrogram_color *= 255
                spectrogram_color = spectrogram_color.astype(int)
                spectrogram_color = np.ascontiguousarray(spectrogram_color, dtype=np.uint8)

                # Save spectrogram as .png
                if save_data: 
                    spectrogram_img = Image.fromarray(spectrogram_color)
                    image_dir = Path(f"{sample_dir}/png/")
                    image_dir.mkdir(parents=True, exist_ok=True)
                    image_path = image_dir / f"{basefilename}.png"
                    spectrogram_img.save(image_path)
                    meta_data["img_file"] = str(image_path)
                    print("Saved image to "+str(image_path))

                # Save metadata as .json 
                meta_data["id"] = spectrogram_id
                file_info["nfft"] = nfft
                meta_data["sample_file"] = file_info
                meta_dir = Path(f"{sample_dir}/metadata/")
                meta_dir.mkdir(parents=True, exist_ok=True)
                json_object = json.dumps(meta_data, indent=4, cls=DtypeEncoder)
                meta_data_path = meta_dir / f"{basefilename}.json"
                with open(meta_data_path, "w") as outfile:
                    outfile.write(json_object)
                print("Saved metadata to "+str(meta_data_path))
        
                # Run inference model
                if not skip_inference:
                    if spectrogram_id > 0: # bug in yolov8, name parameters is broken in predict()
                        model.predictor.save_dir = Path(f"{sample_dir}/predictions/{basefilename}")
                    results = model.predict(source=spectrogram_color[:,:,::-1], conf=0.05, save=True, save_txt=True, save_conf=True, project=f"{sample_dir}/predictions/", name=f"{basefilename}", exist_ok=True)
                    
                spectrogram_id += 1    

        else:
            print("Waiting for sample files to process...")
            time.sleep(wait_time)
            wait_count += 1
            if wait_count == wait_count_limit: 
                print(f"No samples found for {wait_time*wait_count_limit} seconds. Exiting.")
                return

if __name__ == "__main__":
    main()
