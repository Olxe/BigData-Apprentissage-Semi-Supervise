import pandas as pd
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

def fisherScore(dataframe, nbVariable):
    groupByClass = dataframe.groupby(by=[40])
    result = []
    for v in range(0, nbVariable - 1):
        meanClass = np.array(groupByClass[v].mean())
        mean = np.array(dataframe[v].mean())
        effectif = np.array(groupByClass[40].count())

        stdClass = groupByClass[v].std()

        dividende = 0.0
        diviseur = 0.0
        for i in range(0, len(groupByClass)):
            dividende += effectif[i] * (meanClass[i] - mean)**2
            diviseur += effectif[i] * stdClass[i]**2
        
        result.append(dividende/diviseur)
    return np.array(result)
    

# u = np.array(A_LABEL.mean().iloc[:-1])


S1 = fisherScore(A_LABEL, A_LABEL.shape[1])
    
print(S1)

# print("ni")
# print(ni, "\n")
# print("ui")
# print(ui, "\n")

# for key, item in groupByClass:
#     print(groupByClass.get_group(key), "\n\n")
# print("u")
# print(u, "\n")
# print("ecart_type_i")
# print(ecart_type_i, "\n")

# S1_UP = 0
# for i in range(0, c):
#     # print(np.array(ni[i]))
#     # print(np.array(ui[i]))
#     # print(u)
#     S1_UP = S1_UP + (np.multiply(ni[i], (np.subtract(ui[i], u))**2))

# # print(S1_UP)













# test =  np.array((ui - u)**2).T * np.array(ni)
# print(test)



# print(np.array(ni).T)
# S1_UP = 0
# S1_DOWN = 0

# for i in range(1, c):
#     S1_UP = S1_UP + 

# S1 = (ni(ui-u)**2) / (ni*ecart_type_i**2)
# S1 = (ni*ecart_type_i**2)


# print(S1)

# c = 3
# n1 = A_LABEL[A_LABEL[40] == 0]

# tabFirstAverage = []
# tabSecondAverage = []
# tabThirdAverage = []

# # for e in firstClass:
# #     u

# for e in demi_A_labelisee:
#     e[-1]

"""
print(len(A))
print(list(map(lambda x:int(x[n - 1]), A)).count(1))
print(list(map(lambda x:int(x[n - 1]), A)).count(2))
print(list(map(lambda x:int(x[n - 1]), A)).count(3))
"""