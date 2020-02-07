import numpy as np

y=np.random.randint(21, size=(15))
print("list size 20 only: ")
print(y)

x= np.reshape(y,(3, 5))




x[np.where(x==np.max(x,axis=1,keepdims=True))] = 0
print("Maximum value replaced by 0:\n",x)