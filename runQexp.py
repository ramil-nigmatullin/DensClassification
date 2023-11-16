
import numpy as np
import os 
import random
from datetime import datetime
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
import pickle


__all__ = ['QuantumCAChain']


# Define the Quantum Density classification CA model
class QuantumCAChain(CouplingMPOModel, NearestNeighborModel):
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
        Coup = model_params.get('g', 0.)
        gamma = model_params.get('gamma', 0.)
        print('gamma')
        print(gamma)
        print('g')
        print(g)
        
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




## use 'ss' mode to run a large dt simulation to find the steady state
## use 'normal' mode to run a small dt simulation to evaluate the rate of convergence 
## towards the steady state

L = 10        # Number of sites
mode = "normal"

##########################################
## Create a parameter files
param= {
        "dt": 0.01,
        "nsteps": 200,
        "writefreq" : 10,
        "g" : -1,
        "bondCut" : 100,
        "gamma" : 0.6
}

outname = "qexp_params.pkl"
pickle.dump(param, open(outname, 'wb'))



param= {
        "dt": 0.1,
        "nsteps": 200,
        "writefreq": 20,
        "bondCut" : 100,
        "g" : -1,
        "gamma" : 0.6
}



outname = "q_ss_exp_params.pkl"
pickle.dump(param, open(outname, 'wb'))



##########################################
## Load simulation parameters from the parameter file

if (mode == "ss"):
    param = pickle.load(open("q_ss_exp_params.pkl", 'rb'))
    # print(mode)

if (mode == "normal"):
    param = pickle.load(open("qexp_params.pkl", 'rb'))

dt=param['dt']
writefreq=param['writefreq']
nsteps=param['nsteps']
g=param['g']
gamma=param['gamma']
bondCut=param['bondCut']



model_params = dict(L=L, g=g,gamma=gamma,Jz=0,Jxp=0,Jyp=0,Jzp=0,hz=1,hy=0,hx=0, bc_MPS='finite',conserve=None)
M = QuantumCAChain(model_params)  
tebd_params = {
    'order': 1,
#     'delta_tau_list': [1],
     'dt' : dt,
    'N_steps': 1,
    'max_error_E': 1.,
    'trunc_params': {
        'chi_max': bondCut,
        'svd_min': 1.e-12
    },
}

A= np.zeros((L,4))
n, p = 1, .5  # number of trials, probability of each trial
s = np.random.binomial(n, p, int(L/2))



ident = np.array([1.0, 0.0, 0.0, 1.0]);
mesUp = np.array([1.0, 0.0, 0.0, 0.0]);
mesDown = np.array([0.0, 0.0, 0.0, 1.0]);


###################################
#Set up initial conditions
def initializeCA(L):
    #initialize regularly with 10% density
    ups = []

    spacing = round(L/((L/10+1)))
    for j in range(1,round(L/10+1)):
        ups.append(spacing*j)

    return np.array(ups)


ups = initializeCA(L)

for j in range(L): 
    A[j]= mesDown
    for q in range(len(ups)):
        if(j==ups[q]):
            A[j]= mesUp


psi = MPS.from_product_state(M.lat.mps_sites(), A, bc=M.lat.bc_MPS, dtype=complex)
bb = psi._B



###################################
#Set up measurement operators
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


psiIden = MPS.from_product_state(M.lat.mps_sites(), A, bc=M.lat.bc_MPS, dtype=complex)



eng = tebd.TEBDEngine(psi, M, tebd_params)
data = []
data = measurement(eng, mesArr)


if (mode == "normal"):
    with gzip.open('./res/sstate'+str(L)+'.pkl', 'rb') as f:
        sstate = pickle.load(f)
        diff = sstate.copy()


if (mode == "ss"):
    sstate = eng.psi.copy()


counter=0
conv = []
magn = []
entropy = []
bonddim = []
timesteps = []
current_time = datetime.now()


#######################################
# Simulation loop

while eng.evolved_time < nsteps*dt:
# while delta > 0.05:
    timesteps.append(counter)
    eng.run()
    data = measurement(eng, mesArr)
    
    
    magn.append(np.real(data))
    if (mode == "normal"):
        diff = psi.add(sstate,1.0,-1.0)
        conv.append(np.real(diff.overlap(diff)))
    
    entropy.append(psi.entanglement_entropy())
    bonddim.append(psi.chi)

    
    if (counter % writefreq == 0):
        print(counter)
        resdata = {
        "overlap": conv,
        "time" : current_time,
        "timesteps" : timesteps,
        "magnetization" : magn,
        "entropy" : entropy,
        "bond_dimension" : bonddim,
        "bond_max" : bondCut,
        "L": L,
        "dt": dt,
        "nsteps": nsteps,
        "dens": 0.1,
        "g" : g,
        "gamma" : gamma,
        "model" : "quantum",
        "steadystate" : sstate,
        "fstate" : eng.psi
        }

        # Write results to disk
        print("Writing results to disk")

        outname = "./res/CAquantum"+str(current_time)+".pkl"
        pickle.dump(resdata, open(outname, 'wb'))

    counter=counter+1
    # delta = np.sum((abs(mag[round(L/3):round(2*L/3)]-np.mean(mag[round(L/3):round(2*L/3)]))))
    print(counter)
    
print("final bond dimensions: ", psi.chi)

if(mode == 'ss'):
    with gzip.open('./res/sstate'+str(L)+'.pkl', 'wb') as f:
        pickle.dump(eng.psi, f)
