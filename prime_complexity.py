import numpy as np
import matplotlib.pyplot as plt
x0,x1,x2,x3=[],[],[],[]
for i in range(1,40):
    x0.append(np.power(i,2))
    x1.append(np.power(i,1.5))
    x2.append(i*np.log2(i)*np.log2(i))
    x3.append(i)


plt.plot(x3, x3, 'r--', x3, x2, 'b--', x3, x1, 'g--',x3, x0,'y--')
plt.show()