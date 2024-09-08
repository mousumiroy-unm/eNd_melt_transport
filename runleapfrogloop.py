# -*- coding: utf-8 -*-
"""
@author:  Mousumi Roy Feb 2024
"""
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
#import shutil

def tanh_shoulder(Ef0,Efp,w0,shift,t):
    return((1/2)*((Ef0+Efp))+((Efp-Ef0)/2)*np.tanh((t-shift)/w0))
    
def Nd_Mole_Frac_Shoulder(Ef0,Efp,w0,shift,Ichur,t):
    return(1/(((tanh_shoulder(Ef0,Efp,w0,shift,t)/10000)+1)*Ichur+1))

def tanh_pulse(Ef0,Efp,tau,w0,t):
    return(Ef0+((Efp-Ef0)/2)*(np.tanh((t-0.5*tau)/w0) - np.tanh((t-1.5*tau)/w0)))
    
def Nd_Mole_Frac_Input(Ef0,Efp,tau,w0,Ichur,t):
    return(1/(((tanh_pulse(Ef0,Efp,tau,w0,t)/10000)+1)*Ichur+1))
    
## User-defined Parameters
N = 4000 # N, number of grid points in domain length L
L = 1e5 # L, dimensional domain length (m)
tmax = 1e5 # tmax, max dimensional time of run (yrs)
dt = round((5e-7)*tmax,2) # dt, time step -- keep 2 decimals
nout = 100 # nout, number of output files
rho_s = 3300 # rho_s, solid density (kg/m^3)
rho_f = 2800 # rho_f, fluid density (kg/m^3)
alpha = 10 # alpha, transport geometry (10 for channels) - see Hauri(1997a)

d   = 5 #channel spacing (m), used 1,5,10,100 m
phi = 0.1 # volume fraction, used 0.1,0.2
K   = 0.0066 #K1, partition coefficient (maybe list for n numbers of isotopes?), also used 0.025
Df  = 1e-11 # Df1, fluid diffusion coefficient(s) (m^2/s)
Ds  = 1e-11 # Ds1, solid diffusion coefficient(s) (m^2/s)
v   = 1 # v, relative fluid velocity to solid matrix (m/yr)
w0C = 1e3 # w0 for Nd144, pulse change time scale (yrs)
w0f = 1e3 # w0 for Nd143, pulse change time scale (yrs)
shiftC = 5e3 # tau for Nd144, pulse duration (yrs)
shiftf = 5e3 # tau for Nd143, pulse duration (yrs)
dim_or_nondim = 1 # 0 for dim, 1 for nondim

Eps_Nd_s = -2 # starting condition
Eps_Nd_f = -2

Eps_Nd_f_p = 8 # +8 imposed perturbation in the channel 

ICHUR = 0.511836 # assumption
Nd_f_ppm_p_total = 30e-6 # mass fraction ppm, g/1 g rock
Nd_s_ppm_total = 1e-6 # mass fraction ppm, g/1 g rock

Nd_aw = 144.24 # g/mol

# total Nd ppm
Nd_s_mol_m3_total = (Nd_s_ppm_total/Nd_aw)*1000*rho_s

Nd_f_mol_m3_total = Nd_s_mol_m3_total/K # start in eqm

# solid Nd isotopes mol/m^3
Nd_144_s_mol_m3 = Nd_s_mol_m3_total * 0.238
Nd_143_s_mol_m3 = ((Eps_Nd_s/10000)+1)*ICHUR*Nd_144_s_mol_m3

Nd_s_mol_m3 = Nd_144_s_mol_m3 + Nd_143_s_mol_m3 

# fluid Nd isotopes mol/m^3
Nd_144_f_mol_m3 = Nd_f_mol_m3_total * 0.238
Nd_143_f_mol_m3 = ((Eps_Nd_f/10000)+1)*ICHUR*Nd_144_f_mol_m3

Nd_f_mol_m3 = Nd_144_f_mol_m3 + Nd_143_f_mol_m3

# fluid perturbed Nd isotopes mol/m^3
Nd_f_mol_m3_p_total = (Nd_f_ppm_p_total/Nd_aw)*1000*rho_f

Nd_144_f_mol_m3_p = Nd_f_mol_m3_p_total * 0.238
Nd_143_f_mol_m3_p = ((Eps_Nd_f_p/10000)+1)*ICHUR*Nd_144_f_mol_m3_p

Nd_f_mol_m3_p = Nd_144_f_mol_m3_p + Nd_143_f_mol_m3_p

#combined = list(it.product(d,phi,K,Df,Ds,v,w0C,w0f,shiftC,shiftf))

#os.chdir("./OutputFiles")
newpath = "./OutputFiles"
if not os.path.exists(newpath):
    os.makedirs(newpath)
os.chdir(newpath)

#for x in combined:
# convert D's to same units
tau_ex = (d*d)/(alpha*Ds*3.15e7)
tau_ad = (Df*3.15e7)/(v*v)
theta = ((1-phi)*rho_s)/(phi*rho_f)
Da = theta*tau_ad/tau_ex
print(Da)    
theta = (1-phi)*rho_s/(phi*rho_f)
Dahm  = (L/v)*theta*alpha*Ds*3.15e7/(d*d)
print(Dahm)

line1 = "N = " + str(N) + "\n"
line2 = "L = " + str(L) + "\n"
line3 = "tmax = "+ str(tmax) + "\n"
line4 = "dt = " + str(dt) + "\n"
line5 = "nout = " + str(nout) + "\n"
line6 = "rho_s = " + str(rho_s) + "\n"
line7 = "rho_f = " + str(rho_f) + "\n"
line8 = "alpha = " + str(alpha) + "\n"
line9 = "d = " + str(d) + "\n"
line10 = "phi = " + str(phi) + "\n"
line11 = "K = " + str(K) + "\n"
line12 = "Df = " + str(Df) + "\n"
line13 = "Ds = " + str(Ds) + "\n"
line14 = "v = " + str(v) + "\n"
line15 = "w0C = " + str(w0C) + "\n"
line16 = "w0f = " + str(w0f) + "\n"
line17 = "shiftC = " + str(shiftC) + "\n"
line18 = "shiftf = " + str(shiftf) + "\n"
line19 = "Da = " + str(Dahm) + "\n"
line20 = "Cf0 = " + str(Nd_f_mol_m3) + "\n"
line21 = "Cs0 = " + str(Nd_s_mol_m3) + "\n"
line22 = "Cfp = " + str(Nd_f_mol_m3_p) + "\n"
line23 = "ICHUR = " + str(ICHUR) + "\n"
line24 = "Eps_Nd_f = " + str(Eps_Nd_f) + "\n"
line25 = "Eps_Nd_s = " + str(Eps_Nd_s) + "\n"
line26 = "Eps_Nd_f_p = " + str(Eps_Nd_f_p) + "\n"
line27 = "Nd_144_f_0 = " + str(Nd_144_f_mol_m3) + "\n"
line28 = "Nd_143_f_0 = " + str(Nd_143_f_mol_m3) + "\n"
line29 = "Nd_144_s_0 = " + str(Nd_144_s_mol_m3) + "\n"
line30 = "Nd_143_s_0 = " + str(Nd_143_s_mol_m3) + "\n"
line31 = "Nd_144_f_p = " + str(Nd_144_f_mol_m3_p) + "\n"
line32 = "Nd_143_f_p = " + str(Nd_143_f_mol_m3_p) + "\n"
line33 = "dim_or_nondim = " + str(dim_or_nondim) + "\n"

os.remove("Runparams.txt")
inputfile = open("Runparams.txt", "w")

inputfile.write(line1)
inputfile.write(line2)
inputfile.write(line3)
inputfile.write(line4)
inputfile.write(line5)
inputfile.write(line6)
inputfile.write(line7)
inputfile.write(line8)
inputfile.write(line9)
inputfile.write(line10)
inputfile.write(line11)
inputfile.write(line12)
inputfile.write(line13)
inputfile.write(line14)
inputfile.write(line15)
inputfile.write(line16)
inputfile.write(line17)
inputfile.write(line18)
inputfile.write(line19)
inputfile.write(line20)
inputfile.write(line21)
inputfile.write(line22)
inputfile.write(line23)
inputfile.write(line24)
inputfile.write(line25)
inputfile.write(line26)
inputfile.write(line27)
inputfile.write(line28)
inputfile.write(line29)
inputfile.write(line30)
inputfile.write(line31)
inputfile.write(line32)
inputfile.write(line33)

inputfile.close() 

Cs0 = Nd_s_mol_m3
Cf0 = Nd_f_mol_m3
Cfp = Nd_f_mol_m3_p
Nd_144_f_0 = Nd_144_f_mol_m3
Nd_143_f_0 = Nd_143_f_mol_m3
Nd_144_s_0 = Nd_144_s_mol_m3
Nd_143_s_0 = Nd_143_s_mol_m3
Nd_144_f_p = Nd_144_f_mol_m3_p
Nd_143_f_p = Nd_143_f_mol_m3_p

Df = Df * 3.15e7 # m^2/yr
Ds = Ds * 3.15e7 # m^2/yr

## Calculate non-dim Run Parameters
#length_scale = Df/v
length_scale = L
#time_scale = (d*d)/Ds
time_scale = L/v

L = L/length_scale
tmax = tmax/time_scale
dt = dt/time_scale

dx = float(L) / float(N) #distance between spacial nodes
x = np.linspace(0,L,int(N))

Nt = int(tmax/dt)
output_times = np.linspace(0,Nt,int(nout+1))
output_times = [ int(x) for x in output_times ]
output_data = {}

f_f0 = 1/(((Eps_Nd_f/10000)+1)*ICHUR+1) 
f_fp = 1/(((Eps_Nd_f_p/10000)+1)*ICHUR+1)

step = 0
times = np.arange(dt,tmax+dt,dt)
maxiter = len(times)
dtsub = dt*0.5

bc_0_1 = [shiftC/time_scale,w0C/time_scale,0]
bc_0_2 = [shiftf/time_scale,w0f/time_scale,0]

A1 = Dahm
A2 = 1.0
A3 = Df/(length_scale*v)
A4 = A3
B1 = A1/theta
B2 = Ds/(length_scale*v)
B3 = B2

## Initialize Arrays

C_s = Cs0*np.ones(int(N))
C_f = Cf0*np.ones(int(N))

delta_C = K*C_f - C_s

dC_f_dx = np.gradient(C_f,dx)
dC_s_dx = np.gradient(C_s,dx)

d2C_f_dx2 = np.gradient(dC_f_dx,dx)
d2C_s_dx2 = np.gradient(dC_s_dx,dx)

f_f = (Nd_144_f_0/(Nd_144_f_0+Nd_143_f_0))*np.ones(int(N))
f_s = (Nd_144_s_0/(Nd_144_s_0+Nd_143_s_0))*np.ones(int(N))
delta_f = f_f-f_s

df_f_dx = np.gradient(f_f,dx)
df_s_dx = np.gradient(f_s,dx)

d2f_f_dx2 = np.gradient(df_f_dx,dx)
d2f_s_dx2 = np.gradient(df_s_dx,dx)

## Run Leapfrog

for i in range(0,maxiter+1):
    step = step + 1
    t = i*dt
    
# evolve fluid concentrations at int step
    dC_f_dt = - A1 * delta_C - A2 * dC_f_dx + A3 * d2C_f_dx2
    C_f_try = C_f + dC_f_dt * dtsub

# evolve solid concentration at int step
    dC_s_dt = B1 * delta_C + B2 * d2C_s_dx2
    C_s_try = C_s + dC_s_dt * dtsub

# evolve fluid mole fraction at int step
    df_f_dt = - A2 * df_f_dx - A1 * (C_s/C_f) * delta_f + 2 * A3 /C_f * df_f_dx * dC_f_dx + A4 * d2f_f_dx2
    f_f_try = f_f + df_f_dt * dtsub

# evolve solid mole fraction at int step
    df_s_dt =  B1 * K * (C_f/C_s) * delta_f + 2 * B2 /C_s * df_s_dx * dC_s_dx + B3 * d2f_s_dx2
    f_s_try = f_s + df_s_dt * dtsub

# calculate fluid/solid gradients at substep
    dC_f_dx_try = np.gradient(C_f_try, dx)
    dC_s_dx_try = np.gradient(C_s_try, dx)
    d2C_f_dx2_try = np.gradient(dC_f_dx_try, dx)
    d2C_s_dx2_try = np.gradient(dC_s_dx_try, dx)

    df_f_dx_try = np.gradient(f_f_try, dx)
    df_s_dx_try = np.gradient(f_s_try, dx)
    d2f_f_dx2_try = np.gradient(df_f_dx_try, dx)
    d2f_s_dx2_try = np.gradient(df_s_dx_try, dx)

# calculate fluid/solid diffusion and difference terms
    delta_C = K*C_f_try - C_s_try
    delta_f = f_f_try - f_s_try

# use values after intermediate step in order to take next step
    dC_f_dt = - A1 * delta_C - A2 * dC_f_dx_try + A3 * d2C_f_dx2_try
    dC_s_dt = B1 * delta_C + B2 * d2C_s_dx2_try
    df_f_dt = - A2 * df_f_dx_try - A1 * (C_s_try/C_f_try) * delta_f + 2 * A3 /C_f_try * df_f_dx_try * dC_f_dx_try + A4 * d2f_f_dx2_try
    df_s_dt = B1 * K * (C_f_try/C_s_try) * delta_f + 2 * B2 /C_s_try * df_s_dx_try * dC_s_dx_try + B3 * d2f_s_dx2_try

# take full step
    C_f = C_f + dC_f_dt * dt
    C_s = C_s + dC_s_dt * dt
    f_f = f_f + df_f_dt * dt
    f_s = f_s + df_s_dt * dt

# calculate new C arrays
    delta_C = K*C_f-C_s
    delta_f = f_f-f_s

    dC_f_dx = np.gradient(C_f, dx)
    dC_s_dx = np.gradient(C_s,dx)
    d2C_f_dx2 = np.gradient(dC_f_dx,dx)
    d2C_s_dx2 = np.gradient(dC_s_dx,dx)

    df_f_dx = np.gradient(f_f, dx)
    df_s_dx = np.gradient(f_s,dx)
    d2f_f_dx2 = np.gradient(df_f_dx,dx)
    d2f_s_dx2 = np.gradient(df_s_dx,dx)

## Specify Inlet BC for constant Cf
    C_f[0] = tanh_shoulder(Cf0,Cfp,bc_0_1[1],bc_0_1[0],t) 
#        C_f[0] = Cf0 # no concentration gradients, only isotope exchange
    f_f[0] = tanh_shoulder(f_f0,f_fp,bc_0_2[1],bc_0_2[0],t)

## Output at time steps
    if i in output_times:
        if dim_or_nondim == 0:
            output_data[t] = np.vstack((x,C_f,C_s,f_f,f_s)).T
            print(t)
        elif dim_or_nondim == 1:
            output_data[round(t*time_scale,0)] = np.vstack((x*length_scale,C_f,C_s,f_f,f_s)).T
            print(round(t*time_scale,0))
    

## Output data
for key in output_data:
    data = output_data[key]
    np.savetxt(str(key)+'.csv', data, delimiter=",")
    
os.chdir('..')
    
   