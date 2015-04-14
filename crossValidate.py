from __future__ import division
from random import shuffle
import AdaBoost as ab
from datetime import datetime
from math import sqrt

def crossValidate(data, Rds, K, rho=[0]):
    shuffle(data)
    error = [[[0]*len(Rds) for k in range(K)] for r in range(len(rho))]
   
    for indr, r in enumerate(rho):
        for k in range(K):
            with open('status.txt','a') as status:
                print>> status, indr, k, "of", len(rho), K, str(datetime.now())

            train = [line for ind1, line in enumerate(data) if ind1 % K != k]
            test = [line for ind2, line in enumerate(data) if ind2 % K == k]

            error[indr][k] = ab.runAB(train, test, Rds, r)
            #print error[indr][k]

    avgError = [[0]*len(Rds) for i in range(len(rho))]
    stdDev = [[0]*len(Rds) for i in range(len(rho))]
    for i in range(len(rho)):
        for j in range(len(Rds)):
            avgError[i][j] = sum([error[i][k][j] for k in range(K)])/K
            stdDev[i][j] = sqrt(sum([(error[i][k][j]-avgError[i][j])**2 for k in range(K)])/K)
    return avgError, stdDev


#def crossValidateOLD(data, Rds, K):
#    shuffle(data)
#    T = max(Rds)
#    error = [[0]*len(Rds) for k in range(K)]
#
#    for k in range(K): 
#        train = [line for ind1, line in enumerate(data) if ind1 % K != k]
#        test = [line for ind2, line in enumerate(data) if ind2 % K == k] 
#       
#        n = len(train)
#        weights = [1/n]*n
#        alphas = [0]*T
#        divs = [0]*T
#        cols = [0]*T
#        signs = [0]*T
#
#        for i in range(T):
#            divs[i], cols[i], signs[i], alphas[i] = ab.update(train, weights)
#
#            if i+1 in Rds:
#                print k, Rds.index(i+1)
#                error[k][Rds.index(i+1)] = ab.errorsum(divs[:i+1], cols[:i+1], signs[:i+1], alphas[:i+1], test)
#
#    return error


