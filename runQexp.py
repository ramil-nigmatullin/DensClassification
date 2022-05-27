# import numpy as np

# from tenpy.networks.mps import MPS
# from tenpy.models.tf_ising import TFIChain
# from tenpy.algorithms import tebd

# from tenpy.models.model import CouplingMPOModel, NearestNeighborModel, CouplingModel, Model
# from tenpy.tools.params import get_parameter
# from tenpy.networks.site import SpinHalfSite

"""Next-Nearest-neighbour spin-S models.
Uniform lattice of spin-S sites, coupled by next-nearest-neighbour interactions.
We have two variants implementing the same hamiltonian.
The :class:`SpinChainNNN` uses the
:class:`~tenpy.networks.site.GroupedSite` to keep it a
:class:`~tenpy.models.model.NearestNeighborModel` suitable for TEBD,

while the :class:`SpinChainNNN2` just involves longer-range couplings in the MPO.
The latter is preferable for pure DMRG calculations and avoids having to add each of the short
range couplings twice for the grouped sites.
Note that you can also get a :class:`~tenpy.models.model.NearestNeighborModel` for TEBD from the
latter by using :meth:`~tenpy.models.model.MPOModel.group_sites` and
:meth:`~tenpy.models.model.NearestNeighbormodel.from_MPOModel`.
An example for such a case is given in the file ``examples/c_tebd.py``.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np
import os 
import random

import sys

import pickle
import gzip

from tenpy.algorithms import tebd
from tenpy.networks.mps import MPS
from tenpy.linalg.np_conserved import inner
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinSite, GroupedSite
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools.params import asConfig
from tenpy.networks.mps import MPSEnvironment

__all__ = ['SpinChainNNN']


class SpinChainNNN(CouplingMPOModel, NearestNeighborModel):
    r"""Spin-S sites coupled by (next-)nearest neighbour interactions on a `GroupedSite`.
    The Hamiltonian reads:
    .. math ::
        H = \sum_{\langle i,j \rangle, i < j}
                \mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j \\
            + \sum_{\langle \langle i,j \rangle \rangle, i< j}
                \mathtt{Jxp} S^x_i S^x_j + \mathtt{Jyp} S^y_i S^y_j + \mathtt{Jzp} S^z_i S^z_j \\
            - \sum_i
              \mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i
    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbors and
    :math:`\langle \langle i,j \rangle \rangle, i < j` denotes next nearest neighbors.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.
    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinChainNNN` below.
    Options
    -------
    .. cfg:config :: SpinChainNNN
        :include: CouplingMPOModel
        L : int
            Length of the chain in terms of :class:`~tenpy.networks.site.GroupedSite`,
            i.e. we have ``2*L`` spin sites.
        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
        Jx, Jy, Jz, Jxp, Jyp, Jzp, hx, hy, hz : float | array
            Coupling as defined for the Hamiltonian above.
        bc_MPS : {'finite' | 'infinte'}
            MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
    """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        S = model_params.get('S', 0.5)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'),
                                             ('Jxp', 'Jyp'), 'hx', 'hy'], "check Sz conservation"):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        print(conserve)
        spinsite = SpinSite(S, conserve)
        site = GroupedSite([spinsite, spinsite], charges='same')
        return site

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 0.)
        Jy = model_params.get('Jy', 0.)
        Coup=Jx
        gamma = Jy
        

        
        self.add_multi_coupling((-0.75*1j)*gamma,[('Id',[0],0),('Id',[1],0)])
        self.add_multi_coupling((0.5*1j)*gamma,[('Id',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-4.*Coup,[('Sx1',[0],0),('Sx1',[1],0)])
        self.add_multi_coupling(-4.*Coup,[('Sy1',[0],0),('Sy1',[1],0)])
        self.add_multi_coupling((0.5*1j)*gamma,[('Sz1',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(4.*Coup,[('Sx0',[0],0),('Sx0',[1],0)])
        self.add_multi_coupling((2.*1j)*gamma,[('Sx0 Sx1',[0],0),('Sx0 Sx1',[1],0)])
        self.add_multi_coupling((2.*1j)*gamma,[('Sx0 Sy1',[0],0),('Sx0 Sy1',[1],0)])
        self.add_multi_coupling(4.*Coup,[('Sy0',[0],0),('Sy0',[1],0)])
        self.add_multi_coupling((2.*1j)*gamma,[('Sy0 Sx1',[0],0),('Sy0 Sx1',[1],0)])
        self.add_multi_coupling((2.*1j)*gamma,[('Sy0 Sy1',[0],0),('Sy0 Sy1',[1],0)])
        self.add_multi_coupling((0.5*1j)*gamma,[('Sz0',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling((0.5*1j)*gamma,[('Sz0 Sz1',[0],0),('Id',[1],0)])
        self.add_multi_coupling((4.*1j)*gamma,[('Sz0 Sz1',[0],0),('Sz0 Sz1',[1],0)])
        
        
     


def measurement(eng, mesArr):
  
    res = []
    for j in range(len(mesArr)):
        res.append(psi.overlap(mesArr[j]))

    return res


# probs = [0.5, 0.4, 0.2]
# dims = [20, 40, 60,80,100,140,200,300,500,1000]
# dts = [1.0, 2.0, 3.0]
# gammas = [0.8, 1.5, 3.0, 4.0]

# # for ss in range(len(probs)):


# # gamma = 0.8;
# # dt = 1.0;
# # L = 20;
# # g=-2.0;

# python densClassInf.py 10 0.1 0.6 -1.0 1 40 1
# python densClassInf.py 4 0.1 0.6 -1.0 1 20 1


# L = int(sys.argv[1]);
# dt = float(sys.argv[2]);
# gamma = float(sys.argv[3]);
# g = float(sys.argv[4]);
# nUp = float(sys.argv[5]);
# nsteps = float(sys.argv[6]);
# run = sys.argv[7];
# print(g)


Jp=2
# L=15

g=-1.0;
gamma =0.6
dt=0.1
run="1";
nsteps = 200
nUp=1;
UP=1


L = int(sys.argv[1]);
UP = int(sys.argv[2]);
g= -int(sys.argv[3]);





print("finite TEBD, imaginary time evolution, transverse field Ising")
print("L={L:d}, g={g:.2f}".format(L=L, g=g))

model_params = dict(L=L, Jx=g,Jy=gamma,Jz=0,Jxp=0,Jyp=0,Jzp=0,hz=1,hy=0,hx=0, bc_MPS='finite',conserve=None)

M = SpinChainNNN(model_params)  
tebd_params = {
    'order': 1,
#     'delta_tau_list': [1],
     'dt' : dt,
    'N_steps': 1,
    'max_error_E': 1.,
    'trunc_params': {
        'chi_max': 600,
        'svd_min': 1.e-12
    },
}

A= np.zeros((L,4))
n, p = 1, .5  # number of trials, probability of each trial
s = np.random.binomial(n, p, int(L/2))



ident = np.array([1.0, 0.0, 0.0, 1.0]);
mesUp = np.array([1.0, 0.0, 0.0, 0.0]);
mesDown = np.array([0.0, 0.0, 0.0, 1.0]);

# for j in range(L):     
#     A[j]= mesDown

# A[round(L/2)] = mesUp


# a = np.array(random.sample(range(0, round(L/3)), UP))
# ups = np.concatenate((a,a+round(L/3),2*round(L/3)+a))

# for j in range(L): 
#     A[j]= mesDown
#     for q in range(len(ups)):
#         if(j==ups[q]):
#             A[j]= mesUp


ups = np.array(random.sample(range(0, round(L)), round(0.1*L)))

for j in range(L): 
    A[j]= mesDown
    for q in range(len(ups)):
        if(j==ups[q]):
            A[j]= mesUp


print(s)

# for j in range(4):
#     A[j] = np.array([1.0, 0.0, 0.0, 0.0])


# A[4] = np.array([1.0, 0.0, 0.0, 0.0])

psi = MPS.from_product_state(M.lat.mps_sites(), A, bc=M.lat.bc_MPS, dtype=complex)


bb = psi._B

print(bb)


A= np.zeros((L,4))
for j in range(L):
    A[j] = np.array([1.0, 0.0, 0.0, 1.0])


mesArr = []
for k in range(L):
    Q= np.zeros((L,4))
    for j in range(L):
        if(j==k):
            Q[j] = np.array(mesUp) 
        else:
            Q[j] = ident
    mes = MPS.from_product_state(M.lat.mps_sites(), Q, bc=M.lat.bc_MPS, dtype=complex)
    mesArr.append(mes)



Q= np.zeros((L,4))
for j in range(L):
    Q[j] = np.array(mesUp) 

upState = MPS.from_product_state(M.lat.mps_sites(), Q, bc=M.lat.bc_MPS, dtype=complex)

Q= np.zeros((L,4))
for j in range(L):
    Q[j] = np.array(mesDown) 

downState= MPS.from_product_state(M.lat.mps_sites(), Q, bc=M.lat.bc_MPS, dtype=complex)


    


psiIden = MPS.from_product_state(M.lat.mps_sites(), A, bc=M.lat.bc_MPS, dtype=complex)





eng = tebd.TEBDEngine(psi, M, tebd_params)
data = []
data = measurement(eng, mesArr)
trajname = "./trajLoneExcQL"+str(L)+"dt"+str(dt)+"gamma"+str(gamma)+"g"+str(g)+"nUp"+str(int(nUp))+"steps"+str(int(nsteps))+"run"+run+".txt";
bondname = "./bondLoneExcQL"+str(L)+"dt"+str(dt)+"gamma"+str(gamma)+"g"+str(g)+"nUp"+str(int(nUp))+"steps"+str(int(nsteps))+"run"+run+".txt";
entname = "./entropyQ"+str(L)+"dt"+str(dt)+"gamma"+str(gamma)+"g"+str(g)+"nUp"+str(int(nUp))+"steps"+str(int(nsteps))+"run"+run+".txt";


# overlapProbname = "./overlapLoneExcQL"+str(L)+"dt"+str(dt)+"gamma"+str(gamma)+"g"+str(g)+"nUp"+str(int(nUp))+"steps"+str(int(nsteps))+"run"+run+".txt";
overlapProbname = "convergence.txt"

if os.path.exists(entname):
  os.remove(entname)
if os.path.exists(bondname):
  os.remove(bondname)
if os.path.exists(trajname):
  os.remove(trajname)
if os.path.exists(overlapProbname):
  os.remove(overlapProbname)


fEnt = open(entname,'a')
fBond = open(bondname,'a')
# fTraj = open(trajname,'a')
foverlapProb = open(overlapProbname,'a')
np.savetxt(fEnt, psi.entanglement_entropy(),newline=" ")
np.savetxt(fBond, psi.chi,newline=" ")
# np.savetxt(fTraj, np.real(data),newline=" ")

print(psi.overlap(upState))


np.savetxt(foverlapProb, np.array([psi.overlap(upState),psi.overlap(downState)]),newline=" ")


fEnt.write("\n")
fBond.write("\n")
# fTraj.write("\n")
foverlapProb.write("\n")
counter=0
delta =10

# sstate = pickle.load("steadystate")

# infile = open('steadystate','rb')
# sstate = pickle.load(infile)
# infile.close()

with gzip.open('my_data_file.pkl', 'rb') as f:
    sstate = pickle.load(f)

print(sstate)
print(eng.psi)
print(psi.overlap(sstate))

diff = sstate.copy()
while eng.evolved_time < nsteps*dt:
# while delta > 0.05:
    eng.run()
    # print(eng.evolved_time)
    # print(eng.psi.chi)
    data = measurement(eng, mesArr)
    bb = psi._B

    diff = psi.add(sstate,1.0,-1.0)
    np.savetxt(fEnt, psi.entanglement_entropy(),newline=" ")
    np.savetxt(fBond, psi.chi,newline=" ")
    # np.savetxt(fTraj, np.real(data),newline=" ")

    matr = open("./HtrajL"+str(L)+"g"+str(g)+"_"+str(counter)+".txt",'a')
    np.savetxt(matr, np.real(data),delimiter=', ')
    matr.close() 
    np.savetxt(foverlapProb,np.array([diff.overlap(diff)]),newline=" ")
    # np.savetxt(foverlapProb, np.array(psi.overlap(sstate),newline=" ")

    fEnt.write("\n")
    fBond.write("\n")
    counter=counter+1
    mag = np.real(data)
    delta = np.sum((abs(mag[round(L/3):round(2*L/3)]-np.mean(mag[round(L/3):round(2*L/3)]))))
    # print(counter)
    print([counter,diff.overlap(diff)])
    

    # fTraj.write("\n")
    # foverlapProb.write("\n")
    
# print(psi.entanglement_entropy())
# eng.run()  # the main work...
# eng.update(1)
# expectation values
E = np.sum(M.bond_energies(psi))  # M.bond_energies() works only a for NearestNeighborModel
# alternative: directly measure E2 = np.sum(psi.expectation_value(M.H_bond[1:]))
print("E = {E:.13f}".format(E=E))
print("final bond dimensions: ", psi.chi)



