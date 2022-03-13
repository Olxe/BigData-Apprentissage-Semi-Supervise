import random
import numpy as np


# Help data loading
def loadData(filename):
    with open(filename, "r") as file:
        first_line = file.readline()
        tokens = first_line.split()
        matrix = np.loadtxt(filename, usecols=range(len(tokens)))
        return matrix

baseOutput = loadData('Wave.txt') 
random.shuffle(baseOutput)

#Decoupage de nos donnees en 2 bases
A = baseOutput[0:int(len(baseOutput) / 2)]
T = baseOutput[int(len(baseOutput) / 2):]
print(len(A))
print(len(T)) 

#Labelisation de la base A
pourcent = 0.3
demi_A_labelisee = A[0:int(len(A) * pourcent)]
autre_demi_A = A[int(len(A) * pourcent):]
print(len(demi_A_labelisee))
print(len(autre_demi_A))

#Score de fisher
#Recuperation des classes de la base labelisee

firstClass = A[A[:,-1]==0]
secondClass = A[A[:,-1]==1]
thirdClass = A[A[:,-1]==2]

c = 3
n1 = len(firstClass)
n2 = len(secondClass)
n3 = len(thirdClass)

tabFirstAverage = []
tabSecondAverage = []
tabThirdAverage = []

# for e in firstClass:
#     u

print(firstClass)
""" for e in demi_A_labelisee:
    e[-1] """

"""
print(len(A))
print(list(map(lambda x:int(x[n - 1]), A)).count(1))
print(list(map(lambda x:int(x[n - 1]), A)).count(2))
print(list(map(lambda x:int(x[n - 1]), A)).count(3))
"""