#class alto tensor to take input which takes any kind of number which declines the zero values and takes the ones.
#Alto data storage
import os
import sys
import numpy as np
from tensorflow import SparseTensor
import time
from icecream import ic #to yield function and its argument along with result
from joblib import Parallel, delayed, dump, load
import multiprocessing as mp
from tqdm import trange, tqdm

# INPUT PARAMS
# max number of iterations
max_iters = 1000
rank = 16
npartitions = mp.cpu_count()

# size of IType in bits
ITYPE_SIZE = np.uint(64)
# size of FType in bits
FTYPE_SIZE = np.uint(64)
# size of ALTO mask
ALTO_MASK_SIZE = np.uint(64)
MIN_FIBER_REUSE = np.uint(4)
# turn on for more output
ALTO_DEBUG = True

# some global vars
ull1 = np.uint64(1)
ull0 = np.uint64(1)
u1 = np.uint(1)
def clz(num):
    tracker = ull1 << np.uint64(63)
    cnt = np.uint(0)
    while not (num & tracker):
        cnt += np.uint(1)
        tracker = tracker >> ull1
    return cnt
def pdep(src, mask):
    res = np.uint64(0)
    bits = np.uint64(1)
    while mask != 0:
        if src & bits:
            res |= mask & -mask
        mask &= mask - np.uint64(1)
        bits = bits << u1
    return res


def pext(src, mask):
    res = np.uint64(0)
    bits = np.uint64(1)
    while mask != 0:
        if src & mask & -mask:
            res |= bits
        mask &= mask - np.uint64(1)
        bits = bits << u1
    return res


def popcount(num):
    return np.binary_repr(num).count("1")

#Main class AltoTensor for the data storage

class AltoTensor:
    def __init__(self, tft):
        # get basic characteristics
        self.dims = np.array(tft.shape, dtype=np.uint64)
      #nd.dims gets the dimension length
        self.nmode = np.uint64(tft.shape.ndims)
        self.nnz = np.uint64(tft.values.shape[0])
        self.nprtn = np.uint64(npartitions)
        self.prtn_ptr = []
        self.prtn_intervals = [() for _ in range(self.nmode * self.nprtn)]
        self.cr_masks = []
        self.alto_cr_mask = None
        self.vals = np.array(tft.values)
        self.idx = []
        self.mode_masks = []
        self.alto_mask = None

        wtime_s = time.perf_counter()
        self.setup_packed_alto()
        wtime = time.perf_counter() - wtime_s
        print("ALTO setup time = {} (s)".format(wtime))

        # Linearization
        wtime_s = time.perf_counter()
        tft_idx = tft.indices.numpy().astype(np.uint64)
        prts = [int((self.nnz - 1) / npartitions + 1) * i for i in range(npartitions)]
        prts.append(int(self.nnz))
        ## Parallelized linearization
        self.idx = Parallel(
            n_jobs=npartitions, backend="multiprocessing", verbose=1 if ALTO_DEBUG else 0
        )(
            delayed(self.linearize)(tft_idx, self.nmode, self.mode_masks, prts[t], prts[t + 1])
            for t in range(npartitions)
        )
        self.idx = np.concatenate(self.idx).ravel()
        ##Serial linearization
        for i in range(self.nnz):
            alto = np.uint64(0)
            for j in range(self.nmode):
             alto = pdep(tft_idx[i][j],self.mode_masks[j])
            self.idx.append(alto)
            self.idx = np.array(self.idx, dtype=np.uint64)
            wtime = time.perf_counter() - wtime_s
        print("ALTO: Linearization time = {} (s)".format(wtime))

        # Sort the nonzeros based on their line position
        wtime_s = time.perf_counter()
        self.sort_alto()
        wtime = time.perf_counter() - wtime_s
        print("ALTO: sort time = {} (s)".format(wtime))

        # Workload partitioning
        wtime_s = time.perf_counter()
        self.prtn_alto()
        wtime = time.perf_counter() - wtime_s
        print("ALTO: prtn time = {} (s)".format(wtime))

    def linearize(self, old_idx, nmode, mode_masks, sidx, eidx):
        res = np.zeros(eidx - sidx, dtype=np.uint64)
        for idx in range(sidx, eidx):
            alto = np.uint64(0)
            for j in range(nmode):
                alto |= pdep(old_idx[idx][j], mode_masks[j])
            res[sidx - idx] = alto
        return res

    def setup_packed_alto(self):
        alto_bits_min = np.uint(0)
        alto_bits_max = np.uint(0)
        self.alto_mask = np.uint(0)
        max_num_bits = np.uint(0)
        min_num_bits = ITYPE_SIZE
        ALTO_MASK = []
        for n in range(self.nmode):
            ALTO_MASK.append(np.uint64(0))
        mode_bits = [() for _ in range(self.nmode)]
        for n in range(self.nmode):
            mbits = ITYPE_SIZE - clz(self.dims[n])
            mode_bits[n] = (mbits, n)
            alto_bits_min += mbits
            max_num_bits = max(max_num_bits, mbits)
            min_num_bits = min(min_num_bits, mbits)
            print("num_bits for mode-{}={}".format(n + 1, mbits))
        alto_bits_max = max_num_bits * self.nmode
        print("alto_bits_min={}, alto_bits_max={}".format(alto_bits_min, alto_bits_max))
        assert alto_bits_min <= ALTO_MASK_SIZE

        alto_bits = np.uint64(1) << max(np.uint64(3), ITYPE_SIZE - clz(alto_bits_min))
        alto_storage = self.nnz * (FTYPE_SIZE + (alto_bits >> np.uint64(3)))
        print("Alto-power-2 format storage:\t{} MBytes".format(alto_storage / 1000000))

        # only SHORT_FIRST mode implemented
        mode_bits = sorted(mode_bits)
        level = np.uint(0)
        shift = np.uint64(0)
        inc = np.uint(1)
        done = False
        ull1 = np.uint64(1)
        while not done:
            done = True
            for n in range(self.nmode):
                if level < mode_bits[n][0]:
                    ALTO_MASK[mode_bits[n][1]] |= ull1 << shift
                    shift += inc
                    done = False
            level += np.uint(1)
        assert level == (max_num_bits + np.uint(1))

        for n in range(self.nmode):
            self.mode_masks.append(ALTO_MASK[n])
            self.alto_mask |= ALTO_MASK[n]
            ic("ALTO_MASKS[{}] = 0x{}".format(n, np.binary_repr(ALTO_MASK[n])))
        ic("alto_mask = 0x{}".format(np.binary_repr(self.alto_mask)))
#sorting the values into index and values
    def sort_alto(self):
        permutation = self.idx.argsort()
        self.idx = self.idx[permutation]
        self.vals = self.vals[permutation]
#Partitioning the values witht the maximum and minimum values.
    def prtn_par(self, prtid):
        loc_intrvls = []
        fib = []
        for n in range(self.nmode):
            fib.append([self.dims[n], 0])
        for i in range(self.prtn_ptr[prtid], self.prtn_ptr[prtid + 1]):
            alto_idx = self.idx[i]
            for n in range(self.nmode):
                mode_idx = pext(alto_idx, self.mode_masks[n])
                fib[n][0] = min(fib[n][0], mode_idx)
                fib[n][1] = max(fib[n][1], mode_idx)
        for n in range(self.nmode):
            loc_intrvls.append(fib[n])
        return np.array(loc_intrvls)

    def prtn_alto(self):
        nnz_prtn = np.uint((self.nnz + self.nprtn - np.uint(1)) / self.nprtn)
        print("num_prtn={}, nnz_prtn={}".format(self.nprtn, nnz_prtn))

        self.prtn_ptr.append(0)
        for p in range(self.nprtn):
            start_i = np.uint(p * nnz_prtn)
            end_i = start_i + nnz_prtn
            if end_i > self.nnz:
                end_i = self.nnz
            if start_i > end_i:
                start_i = end_i
            self.prtn_ptr.append(end_i)

        ## Parallelized partitioning
        tmp_intervals = Parallel(
            n_jobs=npartitions, backend="multiprocessing", verbose=1 if ALTO_DEBUG else 0
        )(delayed(self.prtn_par)(p) for p in range(self.nprtn))
        self.prtn_intervals = np.array(tmp_intervals).flatten()
        ## Serial partitioning
        for p in range(self.nprtn):
             fib = []
            for n in range(self.nmode):
                 fib.append([self.dims[n], 0])
             for i in range(self.prtn_ptr[p], self.prtn_ptr[p+1]):
                 alto_idx = self.idx[i]
                 for n in range(self.nmode):
                     mode_idx = pext(alto_idx, self.mode_masks[n])
                     fib[n][0] = min(fib[n][0], mode_idx)
                     fib[n][1] = max(fib[n][1], mode_idx)
             for n in range(self.nmode):
                 self.prtn_intervals[np.uint(p * self.nmode + n)] = tuple(fib[n])


def readInTensor(path):
    if os.path.exists(path):
        wtime_s = time.perf_counter()
        with open(path) as f:
            data = f.read().splitlines()
        if data[0] != "sptensor":
            sys.exit(
             
            )
        else:
            # has header
            idx = []
            vals = []
            shape = [int(x) for x in data[2].split()]
            d1_vec = np.array([1 for _ in range(int(data[1]))], dtype=np.double)
            # convert to ints
            for line in range(4, len(data)):
                tmp = np.fromstring(data[line], sep=" ", dtype=np.double)
                # convert from 1-based to 0-based
                idx.append(tmp[:-1].astype(np.uint64) - d1_vec)
                vals.append(tmp[-1])
            del data
            idx = np.array(idx)
            vals = np.array(vals)
            tensor = SparseTensor(indices=idx, values=vals, dense_shape=shape)

            wtime = time.perf_counter() - wtime_s
            ic("Tensor read time = {} (s)".format(wtime))
            return tensor

#Main class to read the file 

def main():
    global npartitions
    try:
        fn = sys.argv[1]
    except IndexError:
        sys.exit("No input tensor given. Exit...\n")
    try:
        npartitions = int(sys.argv[2])
    except IndexError:
        sys.exit(0)
    # read in tensor
    tens = readInTensor(fn)
    print(
        "# Modes\t\t= {}\nRank\t\t= {}\nSparsity\t= {}\nMax iters\t= {}\nDimensions\t= [{}]\nNNZ\t\t= {}".format(
            tens.shape.ndims,
            rank,
            tens.values.shape[0] / tens.shape.num_elements(),
            max_iters,
            " X ".join([str(x) for x in tens.shape]),
            tens.values.shape[0],
        )
    )
    at = AltoTensor(tens)
if __name__ == "__main__":
    ic.configureOutput(prefix="Debug |")
    if not ALTO_DEBUG:
        ic.disable()
    main()
