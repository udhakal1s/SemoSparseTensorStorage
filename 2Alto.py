#This is the second approach to make the ALTO tensor client centric. 

# Still need to add many information if we imnplement the Alto tensor with this process


import os
import sys
import numpy as np
from tensorflow import SparseTensor
import time
# chosse the number of iteration
max_iters = 1500
rank = 3
npartitions = mp.cpu_count()

# size we want to alocate
ITYPE_SIZE = np.uint(32)
FTYPE_SIZE = np.uint(32)

# size of ALTO mask
ALTO_MASK_SIZE = np.uint(32)
MIN_FIBER_REUSE = np.uint(4)
# still need to add some parameter
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
    #main Alto class
class AltoTensor:
   
        #The assert keyword lets you test if a condition in your code returns
        #True, if not, the program will raisean AssertionError. You can write a message
        #to be written if the code returns False
def __init__(self, tensor):
        # defining the all the parameter of Alto tensor
        self.dims = np.array(tensor.shape, dtype=np.uint64)
        self.nmode = np.uint64(tensor.shape.ndims)
        self.nnz = np.uint64(tensor.values.shape[0])
        self.nprtn = np.uint64(npartitions)
        self.prtn_ptr = []
        self.prtn_intervals = [() for _ in range(self.nmode * self.nprtn)]
        self.cr_masks = []
        self.alto_cr_mask = None
        self.vals = np.array(tensor.values)
        self.idx = []
        self.alto_mask = None
        
        assert self.mode_masks = []
        self.mode_pos = []
        assert self.mode_pos

        self.idx = []
        
        assert self.idx
        self.vals = []
        
        assert self.vals
        self.prtn_ptr=[]
        
        assert self.prtn_ptr
        self.prtn_intervals=[]
        
        assert self.prtn_intervals
        self.cr_masks=[0 for i in range nmode]
        
        assert self.cr_masks

        self.prtn_id=[0 for i in range nprtn]
        assert self.prtn_id

        
        self.prtn_masks[0 for i in range nprtn]
        
        assert self.prtn_masks
        self.prtn_mode_masks[0 for i in range nprtn*nmode]
        
        assert self.prtn_mode_masks

        
        ALTO_MASKS[MAX_NUM_MODES]
        for n in range (0 , nmode)
        ALTO_MASKS=self.mode_masks

        for i in range (0,nnz)
        alto=0

        self.vals[i]=spt.vals[i]
        for j in range (0, nmode)
        pass(line 187)

        for j in range(0,nmode)
        mode_idx=0
        pass (line 193)
        assert mpode_idx= spt.cidx(pass)

        ALTO_POS[MAX_NUM_MODES]
        for n in range(o,nmode)
        ALTO_POS=self.mode_pos

        for i in range (0,nnz)
        index=self.idx
        new_index=0
        for n in range(0,nmode)
        pass
        pass 

        self.idx=new_index

        for n in range(0,nmode)
        num_bits=[]
        self.mode_masks=[]
        pass(line 230)
    

        for n in range (0,nmode)
        
       **The iterators that will be used are as follows**
  def linearize(self, old_idx, nmode, mode_masks, sidx, eidx):
        res = np.zeros(eidx - sidx, dtype=np.uint64)
        for idx in range(sidx, eidx):
            alto = np.uint64(0)
            for j in range(nmode):
                alto |= pdep(old_idx[idx][j], mode_masks[j])
            res[sidx - idx] = alto
        return res
  #still need to work on
 
    def setup_packed_alto(self):
    
    
    
    
    
    def sort_alto(self):
        permutation = self.idx.argsort()
        self.idx = self.idx[permutation]
        self.vals = self.vals[permutation]
        
    def __del__:(self):
        del self.dims
        del self.mode_masks
        del self.mode_pos
        del self.idx
        del self.prtn_ptr
        del self.prtn_intervals
        del delf.cr_masks
        del self.prtn_id
        del self.prtn_masks
        del self.prtn_mode_msks
        del(self)

    def __create_da_mem__(

        
#somehow implemented in rough draft but fully not functional
    def set(self, indices, value):
        pass
#somehow implemented but fully not functional
    def get(self, indices):
        pass

    def clear(self):
        pass

    def get_slice(self, starting_indices, ending_indices):
        pass

    def __iter__:


    def __next__:

    
    # another iterator for nnz
    # reading the tensor file
def read(file):
	count=0
	with open(file, 'r') as reader:
		# Create the tensor
		tns = AltoTensor()

		x=0
		for row in reader:

			count=count+1
			#if count % 1000 == 0:
			row = row.split()

			val = float(row.pop())

			idx = [int(i) for i in row]
			x=x+1

			tns.set(idx, val)

	reader.close()
	return tns

def main():
    x = ALtoTensor([4, 5, 3]) # constructpr
    del x # deconstructor
    val = x.get([3,4,6])
    x.set([3,4,6], 45.0)
