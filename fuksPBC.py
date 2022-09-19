
import numpy as np
import sys
import random
import os 
from tenpy.algorithms import tebd
from tenpy.algorithms import network_contractor
from tenpy.networks.mps import MPS
from tenpy.networks.mps import TransferMatrix
from tenpy.linalg.np_conserved import inner
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinSite, GroupedSite
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.tools.params import asConfig
from tenpy.networks.mps import MPSEnvironment

from datetime import datetime
from tenpy.linalg import np_conserved as npc
from random import sample

import pickle

import sys

print(sys.path)

__all__ = ['FuksRuleCA','TransferMatrix']


# Define the Fuks rule CA. The rule is split into four parts:
# one model for partition A of TEBD
# one model for partition B of TEBD
# one model coupling the edge sites of partition A
# one model coupling the edge sites of partition B


class FuksRuleCAv2Edge(CouplingMPOModel, NearestNeighborModel):
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
        site = GroupedSite([spinsite, spinsite,spinsite,spinsite], charges='same')
        return site

    def init_terms(self, model_params):
        Coup = model_params.get('g', 0.)
        gamma = model_params.get('gamma', 0.)
        
        
        coup=np.zeros((1,L-1))
        print(coup[0])
        coup[0][0]=1.0
        hg = coup[0]
        print(hg)
        
        
        self.add_multi_coupling(-0.05*1j*hg,[('Id',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.05*1j*hg,[('Sz3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(0.025*1j*hg,[('Sx2 Sx3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.1*1j*hg,[('Sx2 Sx3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sx2 Sy3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sx2 Sy3',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sy2 Sx3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sy2 Sx3',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(-0.025*1j*hg,[('Sy2 Sy3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1*1j*hg,[('Sy2 Sy3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(0.05*1j*hg,[('Sz2',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(0.05*1j*hg,[('Sz1 Sz3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sz1 Sx2 Sy3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz1 Sx2 Sy3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sz1 Sy2 Sx3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz1 Sy2 Sx3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sz0 Sx2 Sy3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz0 Sx2 Sy3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sz0 Sy2 Sx3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz0 Sy2 Sx3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(0.05*1j*hg,[('Sz0 Sz2',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.1*1j*hg,[('Sz0 Sz1 Sx2 Sx3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.4*1j*hg,[('Sz0 Sz1 Sx2 Sx3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz0 Sz1 Sx2 Sy3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz0 Sz1 Sx2 Sy3',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz0 Sz1 Sy2 Sx3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz0 Sz1 Sy2 Sx3',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(-0.1*1j*hg,[('Sz0 Sz1 Sy2 Sy3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.4*1j*hg,[('Sz0 Sz1 Sy2 Sy3',[0],0),('Sz0 Sz1',[1],0)])




class FuksRuleCAv2(CouplingMPOModel, NearestNeighborModel):
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
        site = GroupedSite([spinsite, spinsite,spinsite,spinsite], charges='same')
        return site

    def init_terms(self, model_params):
        Coup = model_params.get('g', 0.)
        gamma = model_params.get('gamma', 0.)
        

        
        self.add_multi_coupling(-0.05*1j,[('Id',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.05*1j,[('Sz3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(0.025*1j,[('Sx2 Sx3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.1*1j,[('Sx2 Sx3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.025,[('Sx2 Sy3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.025,[('Sx2 Sy3',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(-0.025,[('Sy2 Sx3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.025,[('Sy2 Sx3',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(-0.025*1j,[('Sy2 Sy3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1*1j,[('Sy2 Sy3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(0.05*1j,[('Sz2',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(0.05*1j,[('Sz1 Sz3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.025,[('Sz1 Sx2 Sy3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz1 Sx2 Sy3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.025,[('Sz1 Sy2 Sx3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz1 Sy2 Sx3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.025,[('Sz0 Sx2 Sy3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz0 Sx2 Sy3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.025,[('Sz0 Sy2 Sx3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz0 Sy2 Sx3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(0.05*1j,[('Sz0 Sz2',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.1*1j,[('Sz0 Sz1 Sx2 Sx3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.4*1j,[('Sz0 Sz1 Sx2 Sx3',[0],0),('Sz0 Sz1',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz0 Sz1 Sx2 Sy3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz0 Sz1 Sx2 Sy3',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz0 Sz1 Sy2 Sx3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz0 Sz1 Sy2 Sx3',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(-0.1*1j,[('Sz0 Sz1 Sy2 Sy3',[0],0),('Id',[1],0)])
        self.add_multi_coupling(-0.4*1j,[('Sz0 Sz1 Sy2 Sy3',[0],0),('Sz0 Sz1',[1],0)])



class FuksRuleCAEdge(CouplingMPOModel, NearestNeighborModel):

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
        site = GroupedSite([spinsite, spinsite,spinsite,spinsite], charges='same')
        return site

    def init_terms(self, model_params):
        Coup = model_params.get('g', 0.)
        gamma = model_params.get('gamma', 0.)
        
        coup=np.zeros((1,L-1))
        print(coup[0])
        coup[0][0]=1.0
        hg = coup[0]

        self.add_multi_coupling(-0.05*1j*hg,[('Id',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.05*1j*hg,[('Id',[0],0),('Sz1 Sz3',[1],0)])
        self.add_multi_coupling(0.025*1j*hg,[('Id',[0],0),('Sx0 Sx1',[1],0)])
        self.add_multi_coupling(0.1*1j*hg,[('Id',[0],0),('Sx0 Sx1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Id',[0],0),('Sx0 Sy1 Sz3',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Id',[0],0),('Sx0 Sy1 Sz2',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Id',[0],0),('Sy0 Sx1 Sz3',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Id',[0],0),('Sy0 Sx1 Sz2',[1],0)])
        self.add_multi_coupling(-0.025*1j*hg,[('Id',[0],0),('Sy0 Sy1',[1],0)])
        self.add_multi_coupling(-0.1*1j*hg,[('Id',[0],0),('Sy0 Sy1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(0.05*1j*hg,[('Id',[0],0),('Sz0 Sz2',[1],0)])
        self.add_multi_coupling(0.05*1j*hg,[('Sz3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sz3',[0],0),('Sx0 Sy1',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz3',[0],0),('Sx0 Sy1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sz3',[0],0),('Sy0 Sx1',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz3',[0],0),('Sy0 Sx1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sz2',[0],0),('Sx0 Sy1',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz2',[0],0),('Sx0 Sy1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.025*hg,[('Sz2',[0],0),('Sy0 Sx1',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz2',[0],0),('Sy0 Sx1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(0.05*1j*hg,[('Sz2',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(0.1*1j*hg,[('Sz2 Sz3',[0],0),('Sx0 Sx1',[1],0)])
        self.add_multi_coupling(0.4*1j*hg,[('Sz2 Sz3',[0],0),('Sx0 Sx1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz2 Sz3',[0],0),('Sx0 Sy1 Sz3',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz2 Sz3',[0],0),('Sx0 Sy1 Sz2',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz2 Sz3',[0],0),('Sy0 Sx1 Sz3',[1],0)])
        self.add_multi_coupling(-0.1*hg,[('Sz2 Sz3',[0],0),('Sy0 Sx1 Sz2',[1],0)])
        self.add_multi_coupling(-0.1*1j*hg,[('Sz2 Sz3',[0],0),('Sy0 Sy1',[1],0)])
        self.add_multi_coupling(-0.4*1j*hg,[('Sz2 Sz3',[0],0),('Sy0 Sy1 Sz2 Sz3',[1],0)])




class FuksRuleCA(CouplingMPOModel, NearestNeighborModel):
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
        site = GroupedSite([spinsite, spinsite,spinsite,spinsite], charges='same')
        return site

    def init_terms(self, model_params):
        Coup = model_params.get('g', 0.)
        gamma = model_params.get('gamma', 0.)
          
        self.add_multi_coupling(-0.05*1j,[('Id',[0],0),('Id',[1],0)])
        self.add_multi_coupling(0.05*1j,[('Id',[0],0),('Sz1 Sz3',[1],0)])
        self.add_multi_coupling(0.025*1j,[('Id',[0],0),('Sx0 Sx1',[1],0)])
        self.add_multi_coupling(0.1*1j,[('Id',[0],0),('Sx0 Sx1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.025,[('Id',[0],0),('Sx0 Sy1 Sz3',[1],0)])
        self.add_multi_coupling(-0.025,[('Id',[0],0),('Sx0 Sy1 Sz2',[1],0)])
        self.add_multi_coupling(-0.025,[('Id',[0],0),('Sy0 Sx1 Sz3',[1],0)])
        self.add_multi_coupling(-0.025,[('Id',[0],0),('Sy0 Sx1 Sz2',[1],0)])
        self.add_multi_coupling(-0.025*1j,[('Id',[0],0),('Sy0 Sy1',[1],0)])
        self.add_multi_coupling(-0.1*1j,[('Id',[0],0),('Sy0 Sy1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(0.05*1j,[('Id',[0],0),('Sz0 Sz2',[1],0)])
        self.add_multi_coupling(0.05*1j,[('Sz3',[0],0),('Sz1',[1],0)])
        self.add_multi_coupling(-0.025,[('Sz3',[0],0),('Sx0 Sy1',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz3',[0],0),('Sx0 Sy1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.025,[('Sz3',[0],0),('Sy0 Sx1',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz3',[0],0),('Sy0 Sx1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.025,[('Sz2',[0],0),('Sx0 Sy1',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz2',[0],0),('Sx0 Sy1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.025,[('Sz2',[0],0),('Sy0 Sx1',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz2',[0],0),('Sy0 Sx1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(0.05*1j,[('Sz2',[0],0),('Sz0',[1],0)])
        self.add_multi_coupling(0.1*1j,[('Sz2 Sz3',[0],0),('Sx0 Sx1',[1],0)])
        self.add_multi_coupling(0.4*1j,[('Sz2 Sz3',[0],0),('Sx0 Sx1 Sz2 Sz3',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz2 Sz3',[0],0),('Sx0 Sy1 Sz3',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz2 Sz3',[0],0),('Sx0 Sy1 Sz2',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz2 Sz3',[0],0),('Sy0 Sx1 Sz3',[1],0)])
        self.add_multi_coupling(-0.1,[('Sz2 Sz3',[0],0),('Sy0 Sx1 Sz2',[1],0)])
        self.add_multi_coupling(-0.1*1j,[('Sz2 Sz3',[0],0),('Sy0 Sy1',[1],0)])
        self.add_multi_coupling(-0.4*1j,[('Sz2 Sz3',[0],0),('Sy0 Sy1 Sz2 Sz3',[1],0)])



        
    
def stagger(a,b):
    c = np.zeros(2*len(a))
    for j in range(len(a)):
        c[2*j] = np.real(a[j]);
        c[2*j+1] = np.real(b[j]);
    return c

def measurement(eng, mesArrAll,mesArrDown,psiIden):
  
    res = []
    for j in range(len(mesArrAll)):
       res.append(eng.psi.overlap(mesArrAll[j]))

    
    return res


## Suggested parameter ranges
# dims = [20, 40, 60,80,100,140,200,300,500,1000]
# dts = [1.0, 2.0, 3.0]
# gammas = [0.8, 1.5, 3.0, 4.0]



L = 30
dens = 0.1

bondCut = 400
if(os.path.isfile('./my_counter.pkl') == True):
	loadinit = 1
else:
	loadinit = 0

g=1.0;

gamma = 1.0;
nsteps = 40;
print("finite TEBD, imaginary time evolution, transverse field Ising")
print("L={L:d}, g={g:.2f}".format(L=L, g=g))

model_params = dict(L=L, g=g,gamma=gamma,Jz=0,Jxp=0,Jyp=0,Jzp=0,hz=1,hy=0,hx=0, bc_MPS='finite',conserve=None)

 
dt=2.0
tebd_params = {
    'order': 1,
#     'delta_tau_list': [1],
     'dt' : dt,
    'N_steps': 1,
    'max_error_E': 0.00000000000000000000000000001,
    'trunc_params': {
        'chi_max': bondCut,
        'trunc_cut' : 1.e-12
    },
}

M = FuksRuleCA(model_params) 
Medge = FuksRuleCAEdge(model_params)  
M2 = FuksRuleCAv2(model_params)  
M2edge = FuksRuleCAv2Edge(model_params)  


ident = np.array([1.0, 0.0, 0.0, 1.0]);
mesUp = np.array([1.0, 0.0, 0.0, 0.0]);
mesDown = np.array([0.0, 0.0, 0.0, 1.0]);


mesArrOdd = []
for k in range(L):
    Q= np.zeros((L,16))
    for j in range(L):
        if(j==k):
            Q[j] = np.kron(np.array(mesUp),np.array(ident))           
        else:
            Q[j] = np.kron(ident,ident)                 
    mes = MPS.from_product_state(M.lat.mps_sites(), Q, bc=M.lat.bc_MPS, dtype=complex)
    mesArrOdd.append(mes)



mesArrEven = []
for k in range(L):
    Q= np.zeros((L,16))
    for j in range(L):
        if(j==k):
            Q[j] = np.kron(np.array(ident),np.array(mesUp))
        else:
            Q[j] = np.kron(ident,ident)
    mes = MPS.from_product_state(M.lat.mps_sites(), Q, bc=M.lat.bc_MPS, dtype=complex)
    mesArrEven.append(mes)




mesArrAll = []


Q= np.zeros((L,16))
for j in range(L):
    Q[j] = np.kron(np.array(mesDown),np.array(mesDown))
 

mes = MPS.from_product_state(M.lat.mps_sites(), Q, bc=M.lat.bc_MPS, dtype=complex)
mesArrAll.append(mes)

Q= np.zeros((L,16))
for j in range(L):
    Q[j] = np.kron(np.array(mesUp),np.array(mesUp))
 

mes = MPS.from_product_state(M.lat.mps_sites(), Q, bc=M.lat.bc_MPS, dtype=complex)
mesArrAll.append(mes)



mesArrOddDown = []
for k in range(L):
    Q= np.zeros((L,16))
    for j in range(L):
        if(j==k):
            Q[j] = np.kron(np.array(mesDown),np.array(ident)) 
        else:
            Q[j] = np.kron(ident,ident)
    # print(Q)
    mes = MPS.from_product_state(M.lat.mps_sites(), Q, bc=M.lat.bc_MPS, dtype=complex)
    mesArrOddDown.append(mes)

mesArrEvenDown = []
for k in range(L):
    Q= np.zeros((L,16))
    for j in range(L):
        if(j==k):
            Q[j] = np.kron(np.array(ident),np.array(mesDown)) 
        else:
            Q[j] = np.kron(ident,ident)
    # print(Q)
    mes = MPS.from_product_state(M.lat.mps_sites(), Q, bc=M.lat.bc_MPS, dtype=complex)
    mesArrEvenDown.append(mes)


A= np.zeros((L,16))
for j in range(L):
    A[j] = np.kron(np.array([1.0, 0.0, 0.0, 1.0]),np.array([1.0, 0.0, 0.0, 1.0]))
psiIden = MPS.from_product_state(M.lat.mps_sites(), A, bc=M.lat.bc_MPS, dtype=complex)



A= np.zeros((L,16))



def initializeCA(L):
    #initialize regularly with 10% density
    ups = []
    spacing = round(L/((L/20)))
    first_spacing = round(spacing/2)
    ups.append(first_spacing)
    for j in range(1,round(L/20)):
        ups.append(spacing*j+first_spacing)

    return np.array(ups)

ups = initializeCA(2*L)

print(ups)

for j in range(L): 
    A[j]= np.kron(mesDown,mesDown)
    for q in range(len(ups)):
        if( ups[q] % 2 == 0):
            if(j==round(ups[q]/2)):
                A[j]= np.kron(mesDown,mesUp)
        if( ups[q] % 2 != 0):
            if(j==round(ups[q]/2)):
                A[j]= np.kron(mesUp,mesDown)


  
psi = MPS.from_product_state(M.lat.mps_sites(), A, bc=M.lat.bc_MPS, dtype=complex)
psiInit= psi.copy()



eng = tebd.TEBDEngine(psi, M, tebd_params)
eng2 = tebd.TEBDEngine(psi, M2, tebd_params)
eng2Edge = tebd.TEBDEngine(psi, M2edge, tebd_params)
engEdge = tebd.TEBDEngine(psi, Medge, tebd_params)


data = []
dataDown = []



counter=0



dataEven = measurement(eng, mesArrEven,mesArrEvenDown,psiIden)
dataOdd =  measurement(eng, mesArrOdd,mesArrOddDown,psiIden)
mag = np.real(stagger(dataOdd,dataEven))
# print([len(mag), L])
totalZ=np.sum(mag[:])
print(totalZ)

savedata=np.concatenate((np.array([eng.evolved_time]),mag))
 

matr = open("./MtrajL"+str(L)+"_"+str(counter)+".txt",'w')
np.savetxt(matr, savedata ,delimiter=', ')
matr.close()

matr = open("./UpDownTrajL"+str(L)+"_"+str(counter)+".txt",'w')
dataTest = measurement(eng2, mesArrAll,mesArrAll,psiIden)

np.savetxt(matr, np.real(dataTest),delimiter=', ')
matr.close()
    

counter=counter+1


writefreq = 5

current_time = datetime.now()


overlap = []
magnetization = []
timesteps = []
bonddim = []
# while eng.evolved_time < nsteps*dt:
while counter < 10000:
    
   
    timesteps.append(counter)
    eng.run()
    
    engEdge.psi = eng.psi
  
    # Perform a sequence of swap operations to bring the left and right edge sites next to each other
    for i in reversed(range(0,L-1)):

        siteL, siteR = psi.sites[i], psi.sites[i+1]
        dL, dR = siteL.dim, siteR.dim
        legL, legR = siteL.leg, siteR.leg
        swap_op_dense = np.eye(dL*dR)
        swap_op = npc.Array.from_ndarray(swap_op_dense.reshape([dL, dR, dL, dR]),
                                         [legL, legR, legL.conj(), legR.conj()],
                                         labels=['p1', 'p0', 'p0*', 'p1*'])

        engEdge.psi.swap_sites(i,swap_op)

    # Apply the evolution on the two coupled edge sites

    engEdge.run()

    for i in range(0,L-1):

        siteL, siteR = psi.sites[i], psi.sites[i+1]
        dL, dR = siteL.dim, siteR.dim
        legL, legR = siteL.leg, siteR.leg
        swap_op_dense = np.eye(dL*dR)
        swap_op = npc.Array.from_ndarray(swap_op_dense.reshape([dL, dR, dL, dR]),
                                         [legL, legR, legL.conj(), legR.conj()],
                                         labels=['p1', 'p0', 'p0*', 'p1*'])

        # hmpo.apply_naively(eng.psi)
        engEdge.psi.swap_sites(i,swap_op)  
        # print(eng2.psi.overlap(psiIden))  

    # eng.psi = engEdge.psi
    

    eng2.psi = engEdge.psi
    # # hmpo.apply_naively(eng.psi)
    eng2.run()

    eng2Edge.psi = eng2.psi
  
    # Reverse the previously applied swap operations
    for i in reversed(range(0,L-1)):

        siteL, siteR = psi.sites[i], psi.sites[i+1]
        dL, dR = siteL.dim, siteR.dim
        legL, legR = siteL.leg, siteR.leg
        swap_op_dense = np.eye(dL*dR)
        swap_op = npc.Array.from_ndarray(swap_op_dense.reshape([dL, dR, dL, dR]),
                                         [legL, legR, legL.conj(), legR.conj()],
                                         labels=['p1', 'p0', 'p0*', 'p1*'])

        eng2Edge.psi.swap_sites(i,swap_op)

    eng2Edge.run()


    for i in range(0,L-1):

        siteL, siteR = psi.sites[i], psi.sites[i+1]
        dL, dR = siteL.dim, siteR.dim
        legL, legR = siteL.leg, siteR.leg
        swap_op_dense = np.eye(dL*dR)
        swap_op = npc.Array.from_ndarray(swap_op_dense.reshape([dL, dR, dL, dR]),
                                         [legL, legR, legL.conj(), legR.conj()],
                                         labels=['p1', 'p0', 'p0*', 'p1*'])

        # hmpo.apply_naively(eng.psi)
        eng2Edge.psi.swap_sites(i,swap_op)  
        # print(eng2.psi.overlap(psiIden))  
    eng.psi = eng2Edge.psi


    if(counter % 10 == 0):
    	with open('my_eng.pkl', 'wb') as f:
    		pickle.dump(eng, f)
    	with open('my_eng2.pkl', 'wb') as f:
    		pickle.dump(eng2, f)
    	with open('my_counter.pkl', 'wb') as f:
    		pickle.dump(counter, f)
    
    

    dataEven = measurement(eng2, mesArrEven,mesArrEvenDown,psiIden)
    dataOdd =  measurement(eng2, mesArrOdd,mesArrOddDown,psiIden)
    mag = np.real(stagger(dataOdd,dataEven))

    magnetization.append(mag)
    
    totalZ=np.sum(mag[:])
    print(totalZ)

    
    dataTest = measurement(eng2, mesArrAll,mesArrAll,psiIden)
    overlap.append(dataTest)

    # delta = np.sum((abs(mag[round(L/3):round(2*L/3)]-np.mean(mag[round(L/3):round(2*L/3)]))))
    bonddim.append(psi.chi)

    if (counter % writefreq == 0):
        print(counter)
        resdata = {
        "overlap": overlap,
        "L": L,
        "timesteps" : timesteps,
        "dt": dt,
        "g" : g,
        "nsteps": nsteps,
        "bond_max" : bondCut,        
        "gamma" : gamma,
        "bond_dimension" : bonddim,
        'magnetization' : magnetization,
        "time" : current_time
        }
            # Write results to disk
        print("Writing results to disk")

        outname = "./res/CAfuks"+str(current_time)+".pkl"
        pickle.dump(resdata, open(outname, 'wb'))



    counter = counter+1;
    

E = np.sum(M.bond_energies(psi))  # M.bond_energies() works only a for NearestNeighborModel
# alternative: directly measure E2 = np.sum(psi.expectation_value(M.H_bond[1:]))
print("E = {E:.13f}".format(E=E))
print("final bond dimensions: ", psi.chi)
