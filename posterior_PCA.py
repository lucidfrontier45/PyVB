from sys import argv
import numpy as np
from pynumpdb import readPDBlist, writePDBFile,convCrd
from pynumpdb.anm import _moveCrd
from core import posteriorPCA
import pickle


pdblist_file = argv[1]
vbgmm_model = argv[2]
k = int(argv[3])

pdblist = file(pdblist_file).readlines()
model = pickle.load(file(vbgmm_model))
z = model.z[:,k]
refPDB = pdblist[z.argmax()]
crd,misc,rmsds = readPDBlist(pdblist,refPDB)

eig_val,eig_vec, pcrds  = posteriorPCA(crd,z)
trj = _moveCrd(refPDB,eigval[0],eig_vec[:,0],maxd=eig_val[0])
for i in range(len(trj)):
    log_name = "PDB.%03d" % i
    writePDBFile(log_name,trj[i],misc)
