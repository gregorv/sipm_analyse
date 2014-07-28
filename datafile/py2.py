
import sys
import struct
import numpy as np
import os
import re
import platform
if platform.python_implementation() != "PyPy":
    import yaml

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
    headline = f.readline()
    time_scale = 1
    if headline.startswith("LECROY"):
        time_scale = 1e9
        for i in xrange(4):
            f.readline()
    else:
        f.seek(0)
    time = []
    signal = []
    for l in f:
        l = l.split()
        time.append(float(l[0])*time_scale)
        signal.append(float(l[1]))
    return np.array(time), np.array(signal)


def import_dta(filename, dataset=None):
    with open(filename, "rb") as f:
        header = f.read(20)
        if len(header) < 20:
            raise Exception("Data file broken")
        (m0, m1, m2, m3, m4, version, samples_per_frame,
         num_frames, flags, data_offset) = \
            struct.unpack("cccccBHIBxxxHxx", header)
        if "".join((m0, m1, m2, m3, m4)) != "#DTA\n":
            raise Exception("Invalid magic token")
        if version != 1:
            raise Exception("Unsupported file format version")
        if flags & 1:
            raise Exception("Compression of input files not supported")
        f.seek(data_offset, 1)
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


def import_yaml(filename, dataset=None):
    with open(filename, "rb") as f:
        header = f.read(3)
        if header != "---":
            raise Exception("No YAML data file")
        while data[-4:] != "\n...":
            header += f.read(1)
        print "YAML header", header
        header = yaml.safe_load(header)
        data_offset = f.tell()
        f.seek(0, 1)
        data_size = f.tell() - data_offset
        num_frames = data_size / (2*header["samples_per_frame"]*4)
        print "Num frames", num_frames
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


def get_yaml_header(filename):
    with open(filename, "rb") as f:
        header = f.read(3)
        if header != "---":
            raise Exception("No YAML data file")
        while data[-4:] != "\n...":
            header += f.read(1)
        print "YAML header", header
        header = yaml.safe_load(header)
        data_offset = f.tell()
        f.seek(0, 1)
        data_size = f.tell() - data_offset
        num_frames = data_size / (2*header["samples_per_frame"]*4)
        header["num_frames"] = num_frames
        return header


def get_dta_info(filename):
    with open(filename, "rb") as f:
        header = f.read(20)
        if len(header) < 20:
            raise Exception("Data file broken")
        fmt = "cccccBHIBxxxHxx"
#        print "Struct size", struct.calcsize(fmt)
#        print "BLA", repr(header), len(header)
        (m0, m1, m2, m3, m4, version,
         samples_per_frame, num_frames, flags, data_offset) = \
            struct.unpack(fmt, header)
        if "".join((m0, m1, m2, m3, m4)) != "#DTA\n":
            return None
        else:
            return {"version": version,
                    "samples_per_frame": samples_per_frame,
                    "num_frames": num_frames,
                    "compressed": (flags & 1) != 0,
                    "auto_trigger": (flags & 2) != 0,
                    "data_offset": data_offset,
                    }


def get_dta_userheader(filename):
    with open(filename, "rb") as f:
        header = f.read(20)
        (m0, m1, m2, m3, m4, version, data_offset) = \
            struct.unpack("cccccBxxxxxxxxxxHxx", header)
        if "".join((m0, m1, m2, m3, m4)) != "#DTA\n":
            raise Exception("Not dta file!")
        header_dict = {}
        for line in filter(lambda x: x, f.read(data_offset).split("\n")):
            key, value = map(str.strip, line.split("="))
            if key.startswith("# "):
                key = key[2:]
            if key.endswith("_f"):
                match = re.search("^(-?[0-9]*\.?[0-9]*)(.?)", value)
                try:
                    prefix_value = {"T": 1e12,
                                    "G": 1e9,
                                    "M": 1e6,
                                    "k": 1e3,
                                    "d": 1e-1,
                                    "c": 1e-2,
                                    "m": 1e-3,
                                    "u": 1e-6,
                                    "n": 1e-9,
                                    "p": 1e-12,
                                    "f": 1e-15}[match.group(2)]
                except KeyError:
                    prefix_value = 1.0
                value = float(match.group(1))*prefix_value
            header_dict[key] = value
        return header_dict


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
        return
    print "STEP"
    if len(arg_files) == 1:
        print "Try YAML"
        try:
            header = get_yaml_header(arg_files[0])
            num_frames = header["num_frames"]
            for time, signal, frame_idx in import_yaml(arg_files[0], dataset):
                yield num_frames, frame_idx, time, signal
            return
        except Exception, e:
            print e
    for path in arg_files:
        if os.path.isdir(path):
            directories.append(path)
        else:
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
