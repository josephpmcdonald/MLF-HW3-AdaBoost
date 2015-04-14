from __future__ import division
import numpy as np
import formatData as fd
import AdaBoost as ab
import crossValidate as cv
import matplotlib.pyplot as plt
from datetime import datetime


with open('status.txt','w') as status:
    print>> status, 'Executing AdaBoost and ARho'

print('Formatting data:')
#fd.formatData()
print('Data formatted.')
trainDataNP = np.loadtxt('train.txt')
trainData = trainDataNP.tolist()
testDataNP = np.loadtxt('test.txt')
testData = testDataNP.tolist()

Rds = [1, 100, 200, 500, 1000]
rhoints = range(2,11)
rho = [2**(-i) for i in rhoints]
T = max(Rds)
K = 10

###########################################################################################

with open('status.txt','a') as status:
    print>> status, 'Cross Validation:'

print 'Cross Validation:', K,'folds'
errorRho, stdDevRho = cv.crossValidate(trainData, Rds, K, rho)
print 'A_Rho Cross Validation Complete'#, errorRho
errorAda, stdDevAda = cv.crossValidate(trainData, Rds, K, [0])
print 'AdaBoost Cross Validation Complete'#, errorAda

errorRhoNP = np.array(errorRho)
rMin = rho[np.unravel_index(errorRhoNP.argmin(), errorRhoNP.shape)[0]]
print 'Best Value of rho =', rMin

with open('status.txt','a') as status:
    print>> status, 'Best Value of rho =', rMin
    print>> status, 'Cross Validation Complete at '+str(datetime.now())+'. Computing Test Error for ARho'

############################################################################################

print 'A_rho Test:'
plt.figure(1)
testError = ab.runAB(trainData, testData, Rds, rMin, True)
print 'A_rho Test Error:', testError

with open('status.txt','a') as status:
    print>> status, 'Test Error for ARho complete at '+str(datetime.now())+'. Computing Test Error for AdaBoost'

############################################################################################

print 'AdaBoost Test:'
ABtestError = ab.runAB(trainData, testData, Rds, 0, True)
print 'AdaBoost Test Error:', ABtestError

with open('status.txt','a') as status:
    print>> status, 'Test Error for AdaBoost complete at '+str(datetime.now())+'. Outputting Data!'

plt.title('Cumulative Margins on Training Set')
plt.ylabel('% points with margin below x')
plt.xlabel('x')
plt.legend(loc=4, prop={'size':8})
plt.savefig('Margins.png')
###########################################################################################

with open('ARhoCVError.txt','w') as f:
    print>> f, 'rho in rows =', rho, 'rounds in columns =', Rds
    for row in errorRho:
        print>> f, row
    print>> f, 'std deviations'
    for row1 in stdDevRho:
        print>> f, row1

with open('ABCVError.txt','w') as g:
    print>> g, 'rounds in columns =', Rds
    print>> g, errorAda[0]
    print>> g, 'std deviations'
    print>> g, stdDevAda[0]

with open('ARhoTestError.txt','w') as f1:
    print>> f1, 'rho =', rMin, 'rounds in columns =', Rds
    print>> f1, testError

with open('ABTestError.txt','w') as f2:
    print>> f2, 'rounds in columns =', Rds
    print>> f2, ABtestError

##########################################################################################

plt.figure(3)
for i, line in enumerate(errorRho[:5]):
    plt.errorbar(Rds, line, yerr=stdDevRho[i], label=str(rhoints[:5][i]))
for i, line in enumerate(errorRho[5:]):
    plt.errorbar(Rds, line, yerr=stdDevRho[i], label=str(rhoints[5:][i]), ls='-.')
plt.errorbar(Rds, errorAda[0], yerr=stdDevAda[0], label=str(0), c='k', ls=':')
plt.legend(loc=1, prop={'size':8})
plt.title('Average Cross-Validation Error for K = 10 folds')
plt.xlabel('Number of Boosting Rounds')
plt.savefig('CVError.png')

plt.figure(4)
for i, line in enumerate(errorRho[:5]):
    plt.plot(Rds, line, label=str(rhoints[:5][i]))
for i, line in enumerate(errorRho[5:]):
    plt.plot(Rds, line, label=str(rhoints[5:][i]), ls='-.')
plt.plot(Rds, errorAda[0], label=str(0), c='k', ls=':')
plt.legend(loc=1, prop={'size':8})
plt.title('Average Cross-Validation Error for K = 10 folds')
plt.xlabel('Number of Boosting Rounds')
plt.savefig('CVErrorNoBars.png')

plt.figure(5)
for i, line in enumerate(errorRho[1:5]):
    plt.plot(Rds, line, label=str(rhoints[1:5][i]))
for i, line in enumerate(errorRho[5:]):
    plt.plot(Rds, line, label=str(rhoints[5:][i]), ls='-.')
plt.plot(Rds, errorAda[0], label=str(0), c='k', ls=':')
plt.legend(loc=1, prop={'size':8})
plt.title('Average Cross-Validation Error for K = 10 folds')
plt.xlabel('Number of Boosting Rounds')
plt.savefig('CVErrorNoBarNo2.png')

plt.figure(6)
plt.plot(Rds, testError, label='A_rho')
plt.plot(Rds, ABtestError, label='AdaBst')
plt.title('Test Errors for AB and A_rho, rho ='+str(rMin))
plt.legend(loc=1, prop={'size':8})
plt.xlabel('Number of Boosting Rounds')
plt.savefig('TestError.png')






