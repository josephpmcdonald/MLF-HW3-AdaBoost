from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt

sqrt = math.sqrt
log = math.log
exp = math.exp


def stump(data,weights):
    n = len(data)
    m = len(data[0])

    dividers = [-1.2,-0.6,0.,0.6]
    nd = len(dividers)

    #count error when positive labels are on right(above)/left(below) respectively
    RightEps = [[0 for col in range(m-1)] for row in range(nd)]
    LeftEps = [[0 for col in range(m-1)] for row in range(nd)]
    
    for col in range(1,m):
        for ind, divider in enumerate(dividers):
            divRightEps = 0.
            divLeftEps = 0.
            
            for row in range(n):
                if data[row][col] < divider:
                    #if x[0]=1 add 1, else add 0
                    divRightEps += weights[row]*(1*data[row][0]+1)/2
                    #if x[0]=-1 add 1
                    divLeftEps += weights[row]*(-1*data[row][0]+1)/2
                else:
                    #if x[0]=-1 add 1, else add 0
                    divRightEps += weights[row]*(-1*data[row][0]+1)/2
                    #if x[0]=1 add 1
                    divLeftEps += weights[row]*(1*data[row][0]+1)/2

            RightEps[ind][col-1] = divRightEps
            LeftEps[ind][col-1] = divLeftEps
    
    RightEpsNP = np.array(RightEps)
    LeftEpsNP = np.array(LeftEps)
    RightInd = np.unravel_index(RightEpsNP.argmin(), RightEpsNP.shape)
    LeftInd = np.unravel_index(LeftEpsNP.argmin(), LeftEpsNP.shape)


    """ the method returns the dividing value, the column (coordinate the stump is taken),
    the sign of the labels for points above the threshold, and the error of the base classifier.
    note that 1 is added to the column since Right/LeftInd references the coordinate in the array of
    errors, while the first index of data is a label followed by coordinates"""
    if RightEpsNP[RightInd] < LeftEpsNP[LeftInd]:
        return [dividers[RightInd[0]], RightInd[1]+1, 1, RightEpsNP[RightInd]]
    else:
        return [dividers[LeftInd[0]], LeftInd[1]+1, -1, LeftEpsNP[LeftInd]]


def baseClass(div, sign, lineCol):
    return sign*math.copysign(1, lineCol-div)    


def update(data, weights, rho=0):
    [div, col, sign, eps] = stump(data, weights)
    alpha = 0.5*log((1-eps)/eps)+0.5*log((1-rho)/(1+rho))
    Z = sqrt(eps*(1-eps))*(sqrt((1+rho)/(1-rho))+sqrt((1-rho)/(1+rho)))

    for i, line in enumerate(data):
        weights[i] = weights[i]*exp(-alpha*line[0]*baseClass(div, sign, line[col]))/Z
    
    return [div, col, sign, alpha]


def bcsum(divs, cols, signs, alphas, dataLine):
    T = len(divs)
    if (len(cols) != T | len(signs) != T | len(alphas) != T):
        print 'dimensions don\'t match'
        return
    
    g = 0.

    for i in range(T):
        g += alphas[i]*baseClass(divs[i], signs[i], dataLine[cols[i]])

    return math.copysign(1, g)


def errorsum(divs, cols, signs, alphas, data):
    n = len(data)
    eps = 0

    for line in data:
        eps += (-line[0]*bcsum(divs, cols, signs, alphas, line)+1)/(2*n)

    return eps


def runAB(train, test, Rds, r=0, marg=False):
    n = len(train)
    error = [0]*len(Rds)
    T = max(Rds)

    weights = [1/n]*n
    alphas = [0]*T
    divs = [0]*T
    cols = [0]*T
    signs = [0]*T

    for i in range(T):
        divs[i], cols[i], signs[i], alphas[i] = update(train, weights, r)

        if i+1 in Rds:
            error[Rds.index(i+1)] = errorsum(divs[:i+1], cols[:i+1], signs[:i+1], alphas[:i+1], test)

        # cumulative margins, using training points
        if (i+1 == 500) & marg:
            margins = [0]*n
            for index, line in enumerate(train):
                margins[index] = line[0]*sum([alphas[t]*baseClass(divs[t], signs[t], line[cols[t]]) for t in range(i+1)])/sum([abs(alphas[t]) for t in range(i+1)])

            theta = [x/100 for x in range(101)]
            counts = [sum([1 for m in margins if m <= x])/n for x in theta]
            plt.plot(theta, counts, label=str(r))
#            plt.title('Cumulative Margins on Training Set, rho = '+str(r))
#            plt.ylabel('% points with margin below x')
#            plt.xlabel('x')
#            plt.savefig(plotname)
            
            with open('margins'+str(r)+'.txt','w') as file:
                for m in margins:
                    print>>file, m

    return error


#def stumpNP(data,weights):
#    [n,m]=data.shape
#    dividers = [-1.2,-0.6,0.,0.6]
#    nd = len(dividers)
#
#    #count error when positive labels are on right(above)/left(below) respectively
#    RightEps = np.zeros((nd,m-1))
#    LeftEps = np.zeros((nd,m-1))
#    
#    for col in range(1,m):
#        for ind, divider in enumerate(dividers):
#            divRightEps = 0.
#            divLeftEps = 0.
#            
#            for row in range(n):
#                if data[row,col] < divider:
#                    #if x[0]=1 add 1, else add 0
#                    divRightEps += weights[row]*(1*data[row,0]+1)/2
#                    #if x[0]=-1 add 1
#                    divLeftEps += weights[row]*(-1*data[row,0]+1)/2
#                else:
#                    #if x[0]=-1 add 1, else add 0
#                    divRightEps += weights[row]*(-1*data[row,0]+1)/2
#                    #if x[0]=1 add 1
#                    divLeftEps += weights[row]*(1*data[row,0]+1)/2
#
#            RightEps[ind,col-1] = divRightEps
#            LeftEps[ind,col-1] = divLeftEps
#        
#    RightInd = np.unravel_index(RightEps.argmin(), RightEps.shape)
#    LeftInd = np.unravel_index(LeftEps.argmin(), LeftEps.shape)
#
#
#    """ the method returns the dividing value, the column (coordinate the stump is taken),
#    the sign of the labels for points above the threshold, and the error of the base classifier.
#    note that 1 is added to the column since Right/LeftInd references the coordinate in the array of
#    errors, while the first index of data is a label followed by coordinates"""
#    if RightEps[RightInd] < LeftEps[LeftInd]:
#        return [dividers[RightInd[0]], RightInd[1]+1, 1, RightEps[RightInd]]
#    else:
#        return [dividers[LeftInd[0]], LeftInd[1]+1, -1, LeftEps[LeftInd]]


#def updateRho(data, weights, rho):
#    return update(data, weights, rho)


#def updateOLD(data, weights):
#    [div, col, sign, eps] = stump(data, weights)
#    alpha = 0.5*log((1-eps)/eps)
#    Z = 2*sqrt(eps*(1-eps))
#
#    for ind, line in enumerate(data):
#        weights[ind] = weights[ind]*exp(-alpha*line[0]*baseClass(div, sign, line[col]))/Z
#    
#    return [div, col, sign, alpha]
#
#
#def updateTest(data, weights):
#    [div, col, sign, eps] = stumpNP(data, weights)
#    alpha = 0.5*log((1-eps)/eps)
#    Z = 2*sqrt(eps*(1-eps))
#
#    for ind, line in enumerate(data):
#        weights[ind] = weights[ind]*exp(-alpha*line[0]*baseClass(div, sign, line[col]))/Z
#    
#    return [div, col, sign, alpha]



