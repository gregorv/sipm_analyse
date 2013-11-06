
import sys
import struct
import numpy as np
import os


def import_frame(filename):
    with open(filename, "rb") as f:
        magic = f.read(5)
        if magic == "#BIN\n":
            return import_frame_bin(f)
        elif magic == "#TXT\n":
            return import_frame_txt(f)
        elif magic == "#NPY\n":
            assert("Not implemented!")
        else:
            f.seek(0)
            return import_frame_txt(f)


def import_frame_bin(f):
    data = []
    while True:
        read = f.read(256*4)
        if not len(read):
            break
        data.extend(struct.unpack("f"*256, read))
    assert (len(data) == 4096 or len(data) == 2048),\
        "Expected 4096 or 2048 floats in input file"
    return np.array(data[:len(data)/2]), np.array(data[len(data)/2:])


def import_frame_txt(f):
    time = []
    signal = []
    for l in f:
        l = l.split(" ")
        time.append(float(l[0]))
        signal.append(float(l[1]))
    return np.array(time), np.array(signal)


def import_dta(filename, dataset=None):
    with open(filename, "rb") as f:
        header = f.read(20)
        (version, samples_per_frame, num_frames, compressed) = \
            struct.unpack("xxxxxBHI?xxxxxxx", header)
        if version != 1:
            raise Exception("Unsupported file format version")
        if compressed:
            raise Exception("Compression of input files not supported")
        frame_idx = 0
        if dataset is None:
            for i in xrange(num_frames):
                time = f.read(samples_per_frame*4)
                time = struct.unpack("f"*samples_per_frame, time)
                signal = f.read(samples_per_frame*4)
                signal = struct.unpack("f"*samples_per_frame, signal)
                yield np.array(time), np.array(signal), i
        else:
            for i in sorted(dataset):
                data_pos = 20+samples_per_frame*8*i
                delta_seek = data_pos-f.tell()
                if delta_seek > 0:
                    f.seek(delta_seek, 1)
                time = f.read(samples_per_frame*4)
                time = struct.unpack("f"*samples_per_frame, time)
                signal = f.read(samples_per_frame*4)
                signal = struct.unpack("f"*samples_per_frame, signal)
                yield np.array(time), np.array(signal), i


def get_dta_info(filename):
    with open(filename, "rb") as f:
        header = f.read(20)
        (m0, m1, m2, m3, m4, version, samples_per_frame, num_frames, compressed) = \
            struct.unpack("cccccBHI?xxxxxxx", header)
        if "".join((m0, m1, m2, m3, m4)) != "#DTA\n":
            return None
        else:
            return {"version": version,
                    "samples_per_frame": samples_per_frame,
                    "num_frames": num_frames,
                    "compressed": compressed
                    }


def get_all_data(arg_files, dataset=None):
    num_frames = 0
    frame_idx = 0
    files_to_process = set()
    files_processed = set()
    directories = []

    def discover_files():
        for d in directories:
            for fname in os.listdir(d):
                if fname.endswith(".dat") or fname.endswith(".csv"):
                    fname = os.path.join(d, fname)
                    if fname not in files_processed:
                        files_to_process.add(fname)

    if len(arg_files) == 1 and get_dta_info(arg_files[0]):
        dta_info = get_dta_info(arg_files[0])
        num_frames = dta_info["num_frames"] if dataset is None else len(dataset)
        for time, signal, frame_idx in import_dta(arg_files[0], dataset):
            yield num_frames, frame_idx, time, signal
    else:
        for path in arg_files:
            if os.path.isdir(path):
                directories.append(path)
            elif path.endswith(".dat") or path.endswith(".csv"):
                files_to_process.add(path)
        discover_files()
        while(len(files_to_process)):
            filename = files_to_process.pop()
            files_processed.add(filename)
            if len(files_to_process) == 0:
                discover_files()
            num_frames = len(files_processed) + len(files_to_process)
            frame_idx = len(files_processed)
            #print filename
            time, signal = import_frame(filename)
            yield (num_frames, frame_idx,
                   time, signal)