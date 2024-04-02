import matplotlib.pyplot as P
import numpy as np

t, hsur, hNR = np.load('data.npy') 


fig, ax = P.subplots(figsize=(12, 4))

P.plot(t, hsur, 'b', label='GWSurrogate')
P.plot(t, hNR, 'r--', label='Numerical Relativity')
P.xlabel('Time [Mass]', fontsize=18)
P.ylabel('Strain', fontsize=18)
P.xlim(t[0], 200)
P.legend(frameon=False,fontsize=18)
P.savefig('gwsurrogate.png')
P.close()
