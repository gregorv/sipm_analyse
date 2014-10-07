
import numpy as np

def hasRootSupport():
    try:
        from ROOT import gROOT
        return True
    except ImportError:
        return False

if hasRootSupport():
    from ROOT import TFile, TTree, TBranch, TObjString, TGraph


class EolException(Exception):
    pass

_class_by_extension = {}


class DataFileMeta(type):
    def __init__(self, name, bases, dct):
        global _class_by_extension
        super(DataFileMeta, self).__init__(name, bases, dict)
        if name != "DataFile":
            try:
                _class_by_extension[dct["extension"]] = self
            except KeyError:
                raise AttributeError("Required attribute 'extension' not specified for file-class "+name)


class DataFile:
    __metaclass__ = DataFileMeta

    def frames(self, dataset=None):
        if dataset:
            for frame_idx in sorted(dataset):
                frame = self.next_frame(frame_idx)
                if not frame:
                    break
                yield (len(dataset),) + frame
        else:
            frame = self.next_frame()
            while frame:
                yield (self.num_frames,) + frame
                frame = self.next_frame()


class CsvFile(DataFile):
    extension = ".csv"

    def __init__(self, filename, rw_if_possible=False):
        self.filename = filename
        self.file = None
        self.num_frames = 0
        self.file = open(self.filename)
        self.readHeader()

    def readline_eol(self):
        line = self.file.readline()
        if line == '':
            raise EolException()
        return line

    def __del__(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if self.file:
            self.file.close()

    def readHeader(self):
        self.num_frames = 5

    def next_frame(self, next_idx=None):
        got_correct_frame = False
        try:
            while not got_correct_frame:
                line = self.readline_eol()
                while not line.startswith("##FRAME:"):
                    line = self.readline_eol()
                frame_idx = int(line[8:])
                if next_idx and next_idx != frame_idx:
                    for _ in xrange(1024):
                        self.readline_eol()
                else:
                    got_correct_frame = True
        except EolException:
            return None
        t = np.empty(shape=1024)
        x = np.empty(shape=(1024, 4))
        i = 0
        while line:
            line = self.file.readline()
            if line.startswith("#"):
                continue
            if not line or not line.strip():
                break
            data = map(float, line.split())
            if i > t.shape[0]:
                t.resize((t.shape[0]+1024,))
                x.resize((t.shape[0]+1024, 4))
            t[i] = data[0]
            for j, sample in enumerate(data[1:]):
                if j > 4:
                    pass
                x[i][j] = sample
            i += 1
        return frame_idx, t, x

    def peek_line(self):
        pos = self.file.tell()
        line = self.file.readline()
        self.file.seek(pos)
        return line


class RootFile(DataFile):
    extension = ".root"

    def __init__(self, filename, rw_if_possible=False):
        self.filename = filename
        self.num_frames = 0

        self.file = TFile(filename, "READ")
        if not self.file.IsOpen():
            raise IOError(-1, "Cannot open ROOT file for reading", filename)
        if rw_if_possible:
            self.file.Close()
            self.file = TFile(filename, "UPDATE")
            if not self.file.IsWritable():
                raise IOError(-1, "Cannot open ROOT file in RW mode", filename)
        self.tree = self.file.Get("data")
        self.ch = [TGraph(), TGraph(), TGraph(), TGraph()]
        for i in xrange(1, 5):
            self.tree.SetBranchAddress("ch{0}".format(i), self.ch[i-1])
        self.num_frames = self.tree.GetEntries()
        self.next_frame_idx = 0

    def __del__(self):
        if self.file:
            self.file.Close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if self.file:
            self.file.Close()

    def next_frame(self, next_idx=None):
        if next_idx is not None:
            self.next_frame_idx = next_idx
        frame_idx = self.next_frame_idx
        self.next_frame_idx += 1
        if frame_idx >= self.num_frames:
            return None

        t = None
        x = [None] * 4
        self.tree.GetEntry(frame_idx)
        for ch_num in xrange(4):
            n_points = self.ch[ch_num].GetN()
            if not n_points:
                continue
            if t is None:
                t = np.frombuffer(self.ch[ch_num].GetX(),
                                  dtype=float,
                                  count=n_points)
            x[ch_num] = np.frombuffer(self.ch[ch_num].GetY(),
                                      dtype=float,
                                      count=n_points)
        return frame_idx, t, x

    def getRootFile(self):
        return self.file


def openDatafile(filename, rw_if_possible=True):
    global _class_by_extension
    for ext, cls in _class_by_extension.iteritems():
        if filename.endswith(ext):
            return cls(filename, rw_if_possible)


