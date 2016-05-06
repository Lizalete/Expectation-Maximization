import numpy as np
import scipy.io
import scipy as sp
import scipy.stats as ss
import pylab as py
import math
from numpy import random
from numpy import concatenate
from scipy.optimize import minimize, show_options
from scipy.stats import expon


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def generate_population(mu, N=1000, max_sigma=0.5, mean_sigma=0.08):
    
 """Extract samples from a normal distribution
 with variance distributed as an exponetial distribution
 """
 exp_min_size = 1./max_sigma**2
 exp_mean_size = 1./mean_sigma**2
 sigma = 1/np.sqrt(expon.rvs(loc=exp_min_size, scale=exp_mean_size, size=N))
 #print(np.random.normal(mu, scale=sigma, size=N), sigma)
 #population=np.random.normal(mu, scale=sigma, size=N)
 return np.random.normal(mu, scale=sigma, size=N), sigma
 #plt.plot(population)
 #plt.show()

def pdf_model(x, p):
 mu1, sig1, mu2, sig2,mu3, sig3, pi_1= p
 return pi_1*py.normpdf(x, mu1, sig1) + (1-pi_1)*py.normpdf(x, mu2, sig2) 
 + (1-pi_1)*py.normpdf(x, mu3, sig3)

# generate data from 2 different distributions
N = 340
a = 0.3
b = 0.3
m1 = 0.07 # true mean 1 this is what we want to guess
m2 = 0.025 # true mean 2 this is what we want to guess
m3 = -0.26

mat1 = scipy.io.loadmat('C:/Users/Liz/Desktop/Llantos/LlantoAsfixN.mat', squeeze_me=True, struct_as_record=False)
mat2 = scipy.io.loadmat('C:/Users/Liz/Desktop/Llantos/LlantoNormLyap.mat', squeeze_me=True, struct_as_record=False)
mat3 = scipy.io.loadmat('C:/Users/Liz/Desktop/Llantos/LlantoSordoLyap.mat', squeeze_me=True, struct_as_record=False)

sigM = mat1['PCMtxAsfix']
sigM1 = np.column_stack(sigM)

sigMC = np.concatenate((sigM1[0], sigM1[1], sigM1[2], sigM1[3],
                        sigM1[4], sigM1[5], sigM1[6], sigM1[7],
                        sigM1[8], sigM1[9]),axis=0)
                        
'''                
listS = sigMC.tolist()                        
file = open("sigMC.txt", 'a')
for x in listS:
    file.write(str(x))
    file.write('\n')
file.close()
  '''                      
sigMC11 =sigM1[10]

plt.hist(sigMC11,100)
                        
m1 = np.mean(sigMC)
c1 = np.cov(sigMC)

sigM2 = mat2['MtxNormLyap']
sigM21 = np.column_stack(sigM2)
sigMC2 = np.concatenate((sigM21[0], sigM21[1], sigM21[2], sigM21[3],
                         sigM21[4], sigM21[5], sigM21[6], sigM21[7],
                         sigM21[8], sigM21[9]), axis=0)#sigM2[0]
'''                    
listS2 = sigMC2.tolist()                        
file = open("sigMC2.txt", 'a')
for x in listS2:
    file.write(str(x))
    file.write('\n')
file.close()
'''

sigMC211 = sigM21[10]
plt.hist(sigMC211,100)

m2 =  np.mean(sigMC2)
c2 = np.cov(sigMC2)
                
sigM3 = mat3['MtxSordoLyap']
sigM31 = np.column_stack(sigM3)
sigMC3 = np.concatenate((np.float64(sigM31[0]), np.float64(sigM31[1]), np.float64(sigM31[2]), np.float64(sigM31[3]),
                         np.float64(sigM31[4]), np.float64(sigM31[5]), np.float64(sigM31[6]), np.float64(sigM31[7]), 
                         np.float64(sigM31[8]), np.float64(sigM31[9])),axis=0)#sigM3[0]

'''
listS3 = sigMC3.tolist()                        
file = open("sigMC3.txt", 'a')
for x in listS3:
    file.write(str(x))
    file.write('\n')
file.close()
'''

sigMC311 = sigM31[10]
plt.hist(sigMC311,100)      

m3 = np.mean(sigMC3)
c3 = np.cov(sigMC3)

s1 = sigMC
sig1 = sigMC#generate_population(m1, sigL)

s2 = sigMC2
sig2 = sigMC2#generate_population(m2, N=N*(1-a))

s3 = sigMC3
sig3 = sigMC3#generate_population(m3, N=N*(1-a+b))
s = np.concatenate([s1, s2, s3]) # put all together
sigma_tot = np.concatenate([sig1, sig2, sig3])
'''
py.hist(s, bins=np.r_[-1.5:2:0.025], alpha=0.3, color='g', histtype='stepfilled');
ax = py.twinx(); ax.grid(False)
#py.hist(sigma_tot, 100)
ax.plot(s, 0.1/sigma_tot, 'o', mew=0, ms=6, alpha=0.4, color='b')
py.xlim(-1.5, 1.5); py.title('Sample to be fitted')
py.show()
'''
# Initial guess of parameters and initializations
#p0 = np.array([-0.2, 0.2, 0.8, 0.2, 0.3, 0.3, 0.5])
p0 = np.array([0.07, 0.7, 0.025, 0.23, -0.26, 0.23, 0.5])
mu1, sig1, mu2, sig2, mu3, sig3,pi_1 = p0
mu = np.array([mu1, mu2,mu3]) # estimated means
sig = np.array([sig1, sig2,sig3]) # estimated std dev
pi_ = np.array([pi_1, 1-pi_1, 1-pi_1]) # mixture parameter
gamma = np.zeros((3, s.size))
N = np.zeros(3)
p_new = p0
# EM we start here
#delta = 0.000001
delta = 0.25
improvement = float('inf')
counter = 0

while (improvement > delta):
    # Compute the responsibility func. and new parameters
    for k in [0,1,2]:
        gamma[k,:] = pi_[k]*py.normpdf(s, mu[k], sig[k])/pdf_model(s, p_new) # responsibility
        N[k] = 1.*gamma[k].sum() # effective number of objects to k category
        mu[k] = sum(gamma[k]*s)/N[k] # new sample mean of k category
        sig[k] = np.sqrt( sum(gamma[k]*(s-mu[k])**2)/N[k]) # new sample var of k category
        pi_[k] = N[k]/s.size # new mixture param of k category
    
    # updated parameters will be passed at next iter
    p_old = p_new
    p_new = [mu[0], sig[0], mu[1], sig[1], mu[2], sig[2], pi_[0]]
    # check convergence
    improvement = max(abs(p_old[0] - p_new[0]), abs(p_old[1] - p_new[1]),
                      abs(p_old[2] - p_new[2]))
    counter += 1


'''
    print ("Means: %6.3f %6.3f %6.3f" % (p_new[0], p_new[2], p_new[4]))
    print ("Std dev: %6.3f %6.3f %6.3f" % (p_new[1], p_new[3], p_new[5]))
    print ("Mix (1): %6.3f " % p_new[6])
    print ("Total iterations %d" % counter)
    print (pi_.sum(), N.sum())
'''

#COLUMNA 11
'''
fit = mlab.normpdf(sigMC11,np.mean(sigMC11),np.cov(sigMC11))
py.xlim(-5,5)
py.plot(sigMC11,fit,'ro')
py.hist(sigMC11,100, normed=True)
py.show()

fit = mlab.normpdf(sigMC211,np.mean(sigMC211),np.cov(sigMC211))
py.xlim(-5,5)
py.plot(sigMC211,fit,'ro')
py.hist(sigMC211,100, normed=True)
py.show()

fit = mlab.normpdf(sigMC311,np.mean(sigMC311),np.cov(sigMC311))
py.xlim(-5,5)
py.plot(sigMC311,fit,'ro')
py.hist(sigMC311,100, normed=True)
py.show()

fit = mlab.normpdf(sigMC11,np.mean(sigMC11),np.cov(sigMC11))
py.xlim(-5,5)
py.plot(sigMC11,fit,'ro')
fit2 = mlab.normpdf(sigMC211,np.mean(sigMC211),np.cov(sigMC211))
py.xlim(-5,5)
py.plot(sigMC211,fit2,'bo')
fit3 = mlab.normpdf(sigMC311,np.mean(sigMC311),np.cov(sigMC311))
py.xlim(-5,5)
py.plot(sigMC311,fit3,'go')
py.show()
'''

fit = mlab.normpdf(sigMC,np.mean(sigMC),np.cov(sigMC))
py.plot(sigMC,fit,'ro')
py.hist(sigMC,100, normed=True)
py.show()

fit = mlab.normpdf(sigMC,p_new[0],p_new[1])
py.xlim(-5,5)
py.plot(sigMC,fit,'ro')
py.hist(sigMC,100, normed=True)
py.show()

fit = mlab.normpdf(sigMC2,np.mean(sigMC2),np.cov(sigMC2))
py.xlim(-2,2)
py.plot(sigMC2,fit,'ro')
py.hist(sigMC2,100, normed=True)
py.show()

fit = mlab.normpdf(sigMC2,p_new[2],p_new[3])
py.xlim(-2,2)
py.plot(sigMC2,fit,'ro')
py.hist(sigMC2,100, normed=True)
py.show()


fit = mlab.normpdf(sigMC3,np.mean(sigMC3),np.cov(sigMC3))
py.xlim(-2,2)
py.plot(sigMC3,fit,'ro')
py.hist(sigMC3,100, normed=True)
py.show()

fit = mlab.normpdf(sigMC3,p_new[4],p_new[5])
py.xlim(-2,2)
py.plot(sigMC3,fit,'ro')
py.hist(sigMC3,100, normed=True)
py.show()


fit = mlab.normpdf(sigMC,np.mean(sigMC),np.cov(sigMC))
py.xlim(-5,5)
py.plot(sigMC,fit,'ro')
fit2 = mlab.normpdf(sigMC2,np.mean(sigMC2),np.cov(sigMC2))
py.xlim(-5,5)
py.plot(sigMC2,fit2,'bo')
fit3 = mlab.normpdf(sigMC3,np.mean(sigMC3),np.cov(sigMC3))
py.xlim(-5,5)
py.plot(sigMC3,fit3,'go')
py.show()

fit = mlab.normpdf(sigMC,p_new[0],p_new[1])
py.ylim(0,1)
py.xlim(-2,2)
py.plot(sigMC,fit,'ro')
fit2 = mlab.normpdf(sigMC2,p_new[2],p_new[3])
py.ylim(0,1)
py.xlim(-2,2)
py.plot(sigMC2,fit2,'bo')
fit3 = mlab.normpdf(sigMC3,p_new[4],p_new[5])
py.ylim(0,1)
py.xlim(-2,2)
py.plot(sigMC3,fit3,'go')
py.show()