import matplotlib.pyplot as plt
import numpy as np

x = 10 ** np.linspace(np.log10(2), 100, 1000000)

def k(n):
    return np.log2(n) * np.log2(1 - 2**(- 1. / np.log2(n))) / np.log2(0.75)


plt.plot(x, k(x), label='k')
plt.ylabel('k')
plt.xlabel('N')
plt.xscale('log')
plt.savefig('p1a.pdf', bbox_inches='tight')
