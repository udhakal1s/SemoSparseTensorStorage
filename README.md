# Sparse Tensor Storage Project

The purpose of this application is to allow users to read and import data,
which will be stored using Alto and Hicoo. Hicoo is a form of hierarchial storage.
It will condense the sparse tensors down into ones and zeros that wil be retrieved
as coordinates. Then it will partition those values and coordinates into blocks, 
which are stored into a multidimensional array for fast retrieval. While Alto is a 
adaptive linearized storage format for sparse tensors. It will generate linearized 
indexs that map nonzero values of the high dimensional matrix. Users will be able 
to choose if they want to store data as double, int, or float. The application will 
utilize these python modules and libraries, os and numpy.

Interfaces:
This program will utilize the python programming language. It will be using PyCharm v2021.3.2 and will document all progress and contents on this GitHub Repository. 

Data Structures:
These forms of SparseTensor storage will use multidemensional arrays using the indexs as a method of navigating to a specified location of the array. 
