#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import odeint

#Reading the csv files 
Cov_con=pd.read_csv('time_series_covid19_confirmed_global.csv',header=0,index_col=1)
Cov_death=pd.read_csv('time_series_covid19_deaths_global.csv',header=0,index_col=1)
Cov_rec=pd.read_csv('time_series_covid19_recovered_global.csv',header=0,index_col=1)

# extracting data of India from date: 1st Jan 2021 to 4th Aug 2021
Ind_con=Cov_con.loc['India']['1/1/21':'8/4/21']
Ind_death=Cov_death.loc['India']['1/1/21':'8/4/21']
Ind_rec=Cov_rec.loc['India']['1/1/21':'8/4/21']

#population size
Pop=5.1*sp.Pow(10,7)

#finding total : infected, susceptible and resistant population till date
Infected=Ind_con-Ind_death-Ind_rec
Susceptible=Pop-Ind_con
Resistant=Ind_rec+Ind_death
Susceptible=Susceptible.astype(float)

#finding beta and gamma from data 
elapsed_time=len(Susceptible)
Beta=np.zeros(elapsed_time-1)
for i in range(1,elapsed_time):
    Beta[i-1]=abs(Susceptible[Susceptible.index[i]]-Susceptible[Susceptible.index[i-1]])
    Beta[i-1]/=(Susceptible[Susceptible.index[i-1]]*Infected[Infected.index[i-1]])
Gamma=np.zeros(elapsed_time-1)
for i in range(1,elapsed_time):
    Gamma[i-1]=Resistant[Resistant.index[i]]-Resistant[Resistant.index[i-1]]
    Gamma[i-1]/=Infected[Infected.index[i-1]]
Avg_Beta=np.average(Beta[100:150])
Avg_Gamma=np.average(Gamma[100:150])

#Kermack-McKendrick model
def SIR_model(beta,gamma):
    # function that returns dp/dx
    def SIR(p,t):
        S, I, R = p
        dSdt = -beta*S*I
        dIdt = beta*S*I - gamma*I
        dRdt = gamma*I
        dpdt = [dSdt,dIdt,dRdt]
        return dpdt
    # initial condition [S0, I0, R0]
    p0 = [Susceptible['1/1/21'], Infected['1/1/21'], Resistant['1/1/21']]
    #number of days
    n = elapsed_time
    # time points
    t = np.linspace(0,n,num=n)
    #store solution
    S = np.empty_like(t)
    I = np.empty_like(t)
    R = np.empty_like(t)
    #store initial values
    S[0], I[0], R[0] = p0
    # solve ODE
    for i in range(1,n):
        #span for next step
        tspan = [t[i-1], t[i]]
        #solve for next step
        p = odeint(SIR, p0, tspan)
        #store solution
        S[i], I[i], R[i] = p[1]
        #next initial condition
        p0 = p[1]
    return S,I,R

#Plotting graph of actual covid data and Kermack-McKendrick model
S,I,R=SIR_model(Avg_Beta,Avg_Gamma)
fig,ax = plt.subplots(figsize=(15,8))
Resistant.plot(label='Resisitant')
Infected.plot(label='Infected')
Susceptible.plot(label='Susceptible')
Sus=pd.Series(S,Ind_con.index)
Inf=pd.Series(I,Ind_con.index)
Res=pd.Series(R,Ind_con.index)
Sus.plot(color="green",label="KmK_Sus",linestyle='--')
Inf.plot(color="orange",label="KmK_Inf",linestyle='--')
Res.plot(color="blue",label="KmK_Res",linestyle='--')
plt.xlabel("Date")
plt.ylabel("Population")
plt.legend()
plt.show()
    
#We were able to find the values of Beta and Gamma from the data and Plot
#The Kermack-McKendrick model was almost mateching for Beta=3.569245703074479e-09
# and Gamma = 0.09839973201448345 
