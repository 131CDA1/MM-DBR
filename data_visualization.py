import matplotlib.pyplot as plt
import scipy.io as sio

location = 'Data/MM-DBR/m1/u2_m1_n1.mat'
data = sio.loadmat(location)
print(data['CSIamp'])