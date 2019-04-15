import ctypes
import os
import numpy as np

class BAModel:
    def __init__(self, N, n0, m, L=1, initiate=1, repeats=1, graph="RA"):
        """
        Three models can be run using the Iterate() function in this class:
            1) Barabasi Albert
            2) Pure Random
            3) Existing Vertices
        :param N: The number of total nodes in the graph
        :param n0: The number of initial nodes in the complete initial graph
        :param m: The number of edges added at each time step (not necessarily the degree of each new vertex attached)
        :param initiate: The type of initial graph
        :param repeats: The number of repeats used to improve statistics
        :param graph: The model that is to be used
        """
        self.n0 = n0
        self.N = N
        self.m = m
        self.L = L
        self.initiate=initiate
        self.R = repeats
        self.edges = []
        self.graphtype = graph
        if graph == "BA":
            self.graph = 1
        elif graph == "Pure Random":
            self.graph = 2
        elif graph == "Random Walks":
            self.graph = 3
        else:
            raise Exception('No model exists with that name')
        
    def Iterate(self):
        """runs the model
        objects:
            degrees
            edges
            maxdeg"""
        testlib = ctypes.cdll.LoadLibrary(os.getcwd()+"\BAModel.dll")
        
        self.nodes = np.zeros(self.N,np.int)
        self.nodes_new = np.zeros(self.m,np.int)
        
        max_edge = 10000000
        edge1 = np.zeros(max_edge,np.int)
        edge2 = np.zeros(max_edge,np.int)
        self.degrees = np.zeros(self.R*self.N,np.int)
        self.maxdeg = np.zeros(self.R, np.int)

        testlib.iterations.restype = None
        testlib.iterations(ctypes.c_int(self.N), ctypes.c_int(self.n0), ctypes.c_int(self.m), ctypes.c_int(self.L), ctypes.c_int(self.graph), ctypes.c_int(self.initiate), ctypes.c_int(self.R), np.ctypeslib.as_ctypes(self.nodes), np.ctypeslib.as_ctypes(self.nodes_new), ctypes.c_int(max_edge), np.ctypeslib.as_ctypes(edge1), np.ctypeslib.as_ctypes(edge2), np.ctypeslib.as_ctypes(self.degrees), np.ctypeslib.as_ctypes(self.maxdeg))
        self.degrees = self.degrees[:self.N*self.R]
        self.edges = np.vstack((edge1[:np.argmax(edge1<0)], edge2[:np.argmax(edge1<0)])).T
