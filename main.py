import pandas as pd
import random
import numpy as np
from matplotlib import pyplot as plt 

# Help data loading
def loadData(filename):
    with open(filename, "r") as file:
        first_line = file.readline()
        tokens = first_line.split()
        matrix = np.loadtxt(filename, usecols=range(len(tokens)))
        return matrix

baseOutput = loadData('Wave.txt') 
random.shuffle(baseOutput)

df = pd.DataFrame(baseOutput)
# print(df)

# Decoupage de nos donnees en 2 bases
A = df.iloc[0:int(len(baseOutput) / 2)]
T = df.iloc[int(len(baseOutput) / 2):]

# print(A)
# print(T)

#Labelisation de la base A
amount = 0.3
A_LABEL = A.iloc[0:int(len(A) * amount)]
A_NO_LABEL = A.iloc[int(len(A) * amount):]

# print(A_LABEL)
# print(A_NO_LABEL)

#Score de fisher
#Recuperation des classes de la base labelisée

def scoreDeFisher(dataframe, nbVariable):
    groupByClass = dataframe.groupby(by=[nbVariable - 1])
    effectif = np.array(groupByClass[nbVariable - 1].count())

    result = []
    for v in range(0, nbVariable - 1):
        meanClass = np.array(groupByClass[v].mean())
        mean = np.array(dataframe[v].mean())
        stdClass = groupByClass[v].std()

        dividende = 0.0
        diviseur = 0.0
        for i in range(0, len(groupByClass)):
            dividende += effectif[i] * (meanClass[i] - mean)**2
            diviseur += effectif[i] * stdClass[i]**2
        
        result.append(dividende/diviseur)
    return np.array(result)
    
def scoreDeLaplacien(dataframe, nbVariable):
    matrix = dataframe.to_numpy()
    dividende = 0.0
    result = []

    for v in range(0, nbVariable - 1):
        for rowi in matrix:
            for rowj in matrix:
                vi = rowi[v]
                vj = rowj[v]
                sij = np.exp(- np.dot(rowi-rowj, rowi-rowj) / 0.1)
                dividende += (vi - vj)**2 * sij
        print("End")
        var = dataframe[v].var()
        result.append(dividende/var)
    return np.array(result)
# u = np.array(A_LABEL.mean().iloc[:-1])

S1 = scoreDeFisher(A_LABEL, A_LABEL.shape[1])
# print(S1)

# S2 = scoreDeLaplacien(A_NO_LABEL, A_LABEL.shape[1])
# print(S2)

#Result first iteration
S2 = np.array([1.85463357e-115, 2.30399868e-115, 1.15475623e-114, 9.99346016e-115,
 7.19867478e-115, 7.73226902e-115, 5.98198013e-115, 1.02188809e-114,
 1.21651428e-114, 1.55531165e-114, 1.29177347e-114, 1.56367152e-114,
 1.67976766e-114, 1.43971374e-114, 1.15794927e-114, 1.47839220e-114,
 1.75332745e-114, 2.44694607e-114, 3.56837610e-114, 4.60258324e-114,
 5.64466909e-114, 5.33101298e-114, 5.73612389e-114, 5.93726033e-114,
 6.25081987e-114, 6.46153872e-114, 5.71269408e-114, 6.61460161e-114,
 7.07184640e-114, 7.65325274e-114, 6.72207654e-114, 7.51320356e-114,
 7.76050853e-114, 7.60277606e-114, 7.82074715e-114, 8.65494375e-114,
 8.27104830e-114, 9.14335572e-114, 8.35673300e-114, 9.30703355e-114])

Score = S1 * S2
print(Score)

plt.bar(np.arange(len(Score)),Score)
plt.title("Diagramme à barres") 
plt.show()

ScoreDecroissant = np.sort(Score)[::-1]
# print(ScoreDecroissant)

plt.hist(ScoreDecroissant) 
plt.title("histogram") 
plt.show()

def apprentissage(A, nbVariables, nbClasse):
    poids = []
    n = 100
    e = 0.5

    for _ in range(nbClasse):
        tmp = []
        for __ in range(nbVariables):
            tmp.append(random.random())
        poids.append(tmp) 


    for _ in range(n):
        print(_)
        for x in A:
            classValue = x[-1]
            results = []
            for c in range(nbClasse):
                dotProduct = 0
                for v in range(nbVariables):
                    dotProduct += x[v] * poids[c][v]
                results.append(dotProduct)
            ci = np.argmax(results)
            if ci != classValue:
                for i in range(nbVariables):
                    poids[int(classValue)][i] += e * x[i]
                    poids[ci][i] -= e * x[i]

    return poids

def evaluate(P, x):
    dotProduct = np.dot(P, np.array(x.iloc[:-1]))
    return np.argmax(dotProduct)

def test(P, T):
    compteur = 0

    for _, t in T.iterrows():
        v = evaluate(P, t)
        if v == int(np.array(t)[len(t) - 1]):
            compteur += 1
    print("Accuracy: ", compteur / len(T))

nbVariable = A.shape[1] - 1
nbClasse = len(A.groupby(by=[nbVariable]))

P = apprentissage(np.array(A), nbVariable, nbClasse)

print("Accuracy sans normalisation sur Wave.txt: ")
test(P, T)
