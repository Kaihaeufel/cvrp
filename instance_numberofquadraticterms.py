import networkx as nx
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import matplotlib
import itertools
import cvrplib
from dimod import BinaryQuadraticModel
import dimod as dm
from greedy import SteepestDescentSolver
from dwave.cloud import Client
from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite, DWaveCliqueSampler, LeapHybridSampler
import json
import time as tm
import datetime as dt


n=5
m=5
M=3

def count_variables(n,m,M):
    x=[]
    y=[]
    w=[]

    for v in range(1,n+1):
        y.append(v)

    for v in range(1,n+m-1):
        for j in range(1,m+1):
            x.append((v,j))

    for k in range(1,M+1):
        w.append(k)

    print("Anzahl x_vj: ", len(x))
    print("Anzahl y_v: ", len(y))
    print("Anzahl w_k: ", len(w))
    print("Anzahl Variablen gesamt:", len(x)+len(y)+len(w))

    print("durch Domain Wall spart man dann ", m, "Variablen")

def maximum_original(n,m,M):
    density = n*m + n + M + (n*m+n+M)*(n*m+n+M-1)/2
    return density

def maximum_domain(n,m,M):
    density = (n-1)*m+n+M + ((n-1)*m+n+M)*((n-1)*m+n+M-1)/2
    return density

def maximum_more_variables(n,m,M):
    density = n*m + n + M + (n-1)*m + (n*m+(n-1)*m+n+M)*(n*m+(n-1)*m+n+M-1)/2
    return density

def original_model(n,m,M):

    combinations=[]

    #1 constraint
    for j in range(1, m+1):
        for v in range(1,n+m-1):
            for u in range(v+1,n+m-1):
                zahl = ((v,j),(u,j))
                zahl_alt = ((u,j),(v,j))
                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        combinations.append(zahl)
    print("Nach Nebenbedingung #1")
    print(combinations)
    print(len(combinations))
    print("-----------")

    #2 constraint
    for v in range(1,n+1):
        zahl = (v,v)
        if zahl in combinations:
            pass
        else:
            combinations.append(zahl)
        for j in range(1,m+1):
            azahl = ((v,j), (v,j))
            if azahl in combinations:
                pass
            else:
                combinations.append(azahl)

    for v in range(1,n+1):
        for j in range(1,m+1):
            zahl = (v,(v,j))
            zahl_alt = ((v,j),v)
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    for v in range(1,n+1):
        for j in range(1,m+1):
            for k in range(1,m+1):
                zahl = ((v,j),(v,k))
                zahl_alt = ((v,k),(v,j))
                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        combinations.append(zahl)

    print("Nach Nebenbedingung #2")
    print(combinations)
    print(len(combinations))
    print("-----------")

    for k in range(0,M):
        combinations.append(([k],[k]))
        for v in range(1,n+1):
            zahl = ([k],v)
            if zahl in combinations:
                pass
            else:
                combinations.append(zahl)

    for k in range(0,M):
        for l in range(k,M):
            zahl = ([k],[l])
            zahl_alt = ([l],[k])
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    for v in range(1,n+1):
        for w in range(v,n+1):
            zahl = (v,w)
            zahl_alt = (w,v)
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    print("Nach Nebenbedingung #3")
    print(combinations)
    print(len(combinations))
    print("-----------")

    for u in range(1,n+m-1):
        for v in range(1,n+m-1):
            for j in range(1,m):
                zahl = ((u,j),(v,j+1))
                zahl_alt = ((v,j+1),(u,j))
                if u==v:
                    pass
                elif u > n and v > n:
                    pass
                elif zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        combinations.append(zahl)

    print("Nach Nebenbedingung #4")
    print(combinations)
    print(len(combinations))
    print("-----------")
    return len(combinations)

def domain_wall(n,m,M):

    combinations=[]

    #1 constraint
    for j in range(1, m+1):
        for v in range(1,n+m-3):
            zahl = ((v,j),(v+1,j))
            zahl_alt = ((v+1,j),(v,j))
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)
        einzel = ((1,j),(1,j))
        einzel2 = ((n+m-3,j),(n+m-3,j))
        if einzel in combinations:
            pass
        else:
            combinations.append(einzel)
        if einzel2 in combinations:
            pass
        else:
            combinations.append(einzel2)
    print("Nach Nebenbedingung #1")
    print(combinations)
    print(len(combinations))
    print("-----------")

    #2 constraint
    for v in range(1,n+1):
        zahl = (v,v)
        if zahl in combinations:
            pass
        else:
            combinations.append(zahl)
        for j in range(1,m+1):
            azahl = ((v,j), (v,j))
            if azahl in combinations:
                pass
            else:
                if v == n+m-2:
                    pass
                else:
                    combinations.append(azahl)

    for v in range(1,n+1):
        for j in range(1,m+1):
            zahl = (v,(v,j))
            zahl_alt = ((v,j),v)
            zahl2 = (v,(v-1,j))
            zahl2_alt = ((v-1,j),v)

            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    if v == n+m-2:
                        pass
                    else:
                        combinations.append(zahl)
            if zahl2 in combinations:
                pass
            else:
                if zahl2_alt in combinations:
                    pass
                else:
                    if v-1 == 0:
                        pass
                    else:
                        combinations.append(zahl2)


    for v in range(1,n+1):
        for j in range(1,m+1):
            for k in range(j,m+1):
                zahl = ((v,j),(v,k))
                zahl_alt = ((v,k),(v,j))

                zahl2 = ((v,j), (v-1,k))
                zahl2_alt = ((v-1,k),(v,j))
                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        if v == n+m-2:
                            pass
                        else:
                            combinations.append(zahl)

                # if v == n+m-2:
                #     zahl2 = ((v-1,k),(v-1,k))
                #     zahl2_alt = zahl2
                # if zahl2 in combinations:
                #     pass
                # else:
                #     if zahl2_alt in combinations:
                #         pass
                #     else:
                #         if v-1 == 0:
                #             pass
                #         else:
                #             combinations.append(zahl2)
    for v in range(1, n + 1):
        for j in range(1, m + 1):
            for k in range(1, m + 1):
                zahl = ((v, j), (v-1, k))
                zahl_alt = ((v-1, k), (v, j))

                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        if v == n + m - 2:
                            pass
                        else:
                            if v-1 == 0:
                                pass
                            else:
                                combinations.append(zahl)


    print("Nach Nebenbedingung #2")
    print(combinations)
    print(len(combinations))
    print("-----------")

    for k in range(0,M):
        combinations.append(([k],[k]))
        for v in range(1,n+1):
            zahl = ([k],v)
            if zahl in combinations:
                pass
            else:
                combinations.append(zahl)

    for k in range(0,M):
        for l in range(k,M):
            zahl = ([k],[l])
            zahl_alt = ([l],[k])
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    for v in range(1,n+1):
        for w in range(v,n+1):
            zahl = (v,w)
            zahl_alt = (w,v)
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    print("Nach Nebenbedingung #3")
    print(combinations)
    print(len(combinations))
    print("-----------")

    for u in range(1,n+m-1):
        for v in range(1,n+m-1):
            for j in range(1,m):
                zahl = ((u,j),(v,j+1))
                zahl_alt = ((v,j+1),(u,j))

                zahl2 = ((u,j),(v-1,j+1))
                zahl2_alt = ((v-1,j+1),(u,j))

                zahl3 = ((u-1,j),(v,j+1))
                zahl3_alt = ((v,j+1),(u-1,j))

                zahl4 = ((u-1,j),(v-1,j+1))
                zahl4_alt = ((v-1,j+1),(u-1,j))

                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        if u>n and v>n:
                            pass
                        else:
                            if u == n+m-2:
                                pass
                            else:
                                if v == n+m-2:
                                    pass
                                else:
                                    combinations.append(zahl)

                if zahl2 in combinations:
                    pass
                else:
                    if zahl2_alt in combinations:
                        pass
                    else:
                        if v-1 == 0:
                            pass
                        else:
                            if u>n and v-1>n:
                                pass
                            else:
                                if u == n + m - 2:
                                    pass
                                else:
                                    combinations.append(zahl2)

                if zahl3 in combinations:
                    pass
                else:
                    if zahl3_alt in combinations:
                        pass
                    else:
                        if u-1 ==0:
                            pass
                        else:
                            if u-1>n and v>n:
                                pass
                            else:
                                if v == n + m - 2:
                                    pass
                                else:
                                    combinations.append(zahl3)

                if zahl4 in combinations:
                    pass
                else:
                    if zahl4_alt in combinations:
                        pass
                    else:
                        if u-1 == 0:
                            pass
                        else:
                            if v-1 == 0:
                                pass
                            else:
                                if u-1 >n and v-1>n:
                                    pass
                                else:
                                    combinations.append(zahl4)
    print("Nach Nebenbedingung #4")
    print(combinations)
    print(len(combinations))
    print("-----------")
    return len(combinations)

def domain_and_reduced_y(n,m,M):

    combinations = []

    #1 constraint
    for j in range(1, m+1):
        for v in range(1,n+m-3):
            zahl = ((v,j),(v+1,j))
            zahl_alt = ((v+1,j),(v,j))
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)
        einzel = ((1,j),(1,j))
        einzel2 = ((n+m-3,j),(n+m-3,j))
        if einzel in combinations:
            pass
        else:
            combinations.append(einzel)
        if einzel2 in combinations:
            pass
        else:
            combinations.append(einzel2)
    # print("Nach Nebenbedingung #1")
    # print(combinations)
    # print(len(combinations))
    # print("-----------")

    # 2 constraint
    for v in range(1,n+1):
        for i in range(1,m+1):
            for j in range(1,m+1):
                zahl = ((v, i), (v, j))
                zahl_alt = ((v, j), (v,i))

                zahl2 = ((v,i), (v - 1, j))
                zahl2_alt = ((v - 1, j), (v,i))

                zahl3 = ((v - 1, i), (v, j))
                zahl3_alt = ((v, j), (v - 1, i))

                zahl4 = ((v - 1, i), (v - 1, j))
                zahl4_alt = ((v - 1, j), (v - 1, i))

                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        if v==n+m-2:
                            pass
                        else:
                            combinations.append(zahl)

                if zahl2 in combinations:
                    pass
                else:
                    if zahl2_alt in combinations:
                        pass
                    else:
                        if v - 1 == 0:
                            pass
                        else:
                            if v==n+m-2:
                                pass
                            else:
                                combinations.append(zahl2)

                if zahl3 in combinations:
                    pass
                else:
                    if zahl3_alt in combinations:
                        pass
                    else:
                        if v - 1 == 0:
                            pass
                        else:
                            if v==n+m-2:
                                pass
                            else:
                                combinations.append(zahl3)

                if zahl4 in combinations:
                    pass
                else:
                    if zahl4_alt in combinations:
                        pass
                    else:
                        if v - 1 == 0:
                            pass
                        else:
                            combinations.append(zahl4)

    # print("Nach Nebenbedingung #2")
    # print(combinations)
    # print(len(combinations))
    # print("-----------")

    for k in range(0, M):
        combinations.append(([k], [k]))

    for k in range(0, M):
        for l in range(k, M):
            zahl = ([k], [l])
            zahl_alt = ([l], [k])
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    for v in range(1,n+1):
        for u in range(1,n+1):
            for j in range(1,m+1):
                for i in range(1,m+1):
                    zahl = ((v,j),(u,i))
                    zahl_alt = ((u,i), (v,j))

                    if zahl in combinations:
                        pass
                    else:
                        if zahl_alt in combinations:
                            pass
                        else:
                            if v == n+m-2:
                                pass
                            elif u == n+m-2:
                                pass
                            else:
                                combinations.append(zahl)

    for k in range(0,M):
        for v in range(1,n+1):
            for j in range(1,m+1):
                zahl = ([k], (v,j))

                if zahl in combinations:
                    pass
                else:
                    if v == n+m-2:
                        pass
                    else:
                        combinations.append(zahl)



    # print("Nach Nebenbedingung #3")
    # print(combinations)
    # print(len(combinations))
    # print("-----------")

    for u in range(1, n + m - 1):
        for v in range(1, n + m - 1):
            for j in range(1, m):
                zahl = ((u, j), (v, j + 1))
                zahl_alt = ((v, j + 1), (u, j))

                zahl2 = ((u, j), (v - 1, j + 1))
                zahl2_alt = ((v - 1, j + 1), (u, j))

                zahl3 = ((u - 1, j), (v, j + 1))
                zahl3_alt = ((v, j + 1), (u - 1, j))

                zahl4 = ((u - 1, j), (v - 1, j + 1))
                zahl4_alt = ((v - 1, j + 1), (u - 1, j))

                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        if u > n and v > n:
                            pass
                        else:
                            if u == n + m - 2:
                                pass
                            else:
                                if v == n + m - 2:
                                    pass
                                else:
                                    combinations.append(zahl)

                if zahl2 in combinations:
                    pass
                else:
                    if zahl2_alt in combinations:
                        pass
                    else:
                        if v - 1 == 0:
                            pass
                        else:
                            if u > n and v - 1 > n:
                                pass
                            else:
                                if u == n + m - 2:
                                    pass
                                else:
                                    combinations.append(zahl2)

                if zahl3 in combinations:
                    pass
                else:
                    if zahl3_alt in combinations:
                        pass
                    else:
                        if u - 1 == 0:
                            pass
                        else:
                            if u - 1 > n and v > n:
                                pass
                            else:
                                if v == n + m - 2:
                                    pass
                                else:
                                    combinations.append(zahl3)

                if zahl4 in combinations:
                    pass
                else:
                    if zahl4_alt in combinations:
                        pass
                    else:
                        if u - 1 == 0:
                            pass
                        else:
                            if v - 1 == 0:
                                pass
                            else:
                                if u - 1 > n and v - 1 > n:
                                    pass
                                else:
                                    combinations.append(zahl4)
    # print("Nach Nebenbedingung #4")
    # print(combinations)
    # print(len(combinations))
    # print("-----------")
    return len(combinations)

def reduced(n,m,M):

    combinations=[]

    #1 constraint
    for j in range(1, m+1):
        for v in range(1,n+m-1):
            for u in range(v+1,n+m-1):
                zahl = ((v,j),(u,j))
                zahl_alt = ((u,j),(v,j))
                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        combinations.append(zahl)
    # print("Nach Nebenbedingung #1")
    # print(combinations)
    # print(len(combinations))
    # print("-----------")

    #2 constraint
    for v in range(1,n+1):
        for i in range(1,m+1):
            for j in range(1,m+1):
                zahl = ((v,i),(v,j))
                zahl_alt = ((v,j),(v,i))
                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        combinations.append(zahl)

    # print("Nach Nebenbedingung #2")
    # print(combinations)
    # print(len(combinations))
    # print("-----------")

    #3 constraint
    for k in range(0,M):
        combinations.append(([k],[k]))
        for v in range(1,n+1):
            for j in range(1,m+1):
                zahl = ([k],(v,j))
                if zahl in combinations:
                    pass
                else:
                    combinations.append(zahl)

    for k in range(0,M):
        for l in range(k,M):
            zahl = ([k],[l])
            zahl_alt = ([l],[k])
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    for u in range(1,n+1):
        for v in range(1,n+1):
            for i in range(1,m+1):
                for j in range(1,m+1):
                    zahl = ((u,i), (v,j))
                    zahl_alt = ((v,j),(u,i))
                    if zahl in combinations:
                        pass
                    else:
                        if zahl_alt in combinations:
                            pass
                        else:
                            combinations.append(zahl)

    # print("Nach Nebenbedingung #3")
    # print(combinations)
    # print(len(combinations))
    # print("-----------")

    for u in range(1,n+m-1):
        for v in range(1,n+m-1):
            for j in range(1,m):
                zahl = ((u,j),(v,j+1))
                zahl_alt = ((v,j+1),(u,j))
                if u==v:
                    pass
                elif u > n and v > n:
                    pass
                elif zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        combinations.append(zahl)

    # print("Nach Nebenbedingung #4")
    # print(combinations)
    # print(len(combinations))
    # print("-----------")
    return len(combinations)

def more_variables(n,m,M):
    combinations=[]


    #0 constraint
    for v in range(1,n+1):
        for j in range(1,m+1):
            zahl = ((24,v,j), (v,j))
            zahl2 = ((24,v,j), (v-1,j))
            zahl3 = ((v,j), (v-1,j))
            zahl4 = ((24,v,j),(24,v,j))
            zahl5 = ((v,j),(v,j))

            if v == n + m - 2:
                zahl = ((24, v, j), (24, v, j))
                zahl5 =((24, v, j), (24, v, j))
            if v-1 == 0:
                zahl2 = ((24, v, j), (24, v, j))
                zahl3 = ((v,j),(v,j))
            if zahl in combinations:
                pass
            else:
                combinations.append(zahl)
            if zahl2 in combinations:
                pass
            else:
                combinations.append(zahl2)
            if zahl3 in combinations:
                pass
            else:
                combinations.append(zahl3)
            if zahl4 in combinations:
                pass
            else:
                combinations.append(zahl4)
            if zahl5 in combinations:
                pass
            else:
                combinations.append(zahl5)
    print("Nach Nebenbedingung #0")
    print(combinations)
    print(len(combinations))
    print("-----------")
    #1 constraint
    for j in range(1, m+1):
        for v in range(1,n+m-3):
            zahl = ((v,j),(v+1,j))
            zahl_alt = ((v+1,j),(v,j))
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)
        einzel = ((1,j),(1,j))
        einzel2 = ((n+m-3,j),(n+m-3,j))
        if einzel in combinations:
            pass
        else:
            combinations.append(einzel)
        if einzel2 in combinations:
            pass
        else:
            combinations.append(einzel2)
    print("Nach Nebenbedingung #1")
    print(combinations)
    print(len(combinations))
    print("-----------")

    #2 constraint
    for v in range(1,n+1):
        zahl = (v,v)
        if zahl in combinations:
            pass
        else:
            combinations.append(zahl)
        for j in range(1,m+1):
            azahl = ((24,v,j), (24,v,j))
            if azahl in combinations:
                pass
            else:
                combinations.append(azahl)

    for v in range(1,n+1):
        for j in range(1,m+1):
            zahl = (v,(24,v,j))
            zahl_alt = ((24,v,j),v)
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    for v in range(1,n+1):
        for j in range(1,m+1):
            for k in range(1,m+1):
                zahl = ((24,v,j),(24,v,k))
                zahl_alt = ((24,v,k),(24,v,j))
                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        combinations.append(zahl)

    print("Nach Nebenbedingung #2")
    print(combinations)
    print(len(combinations))
    print("-----------")

    for k in range(0,M):
        combinations.append(([k],[k]))
        for v in range(1,n+1):
            zahl = ([k],v)
            if zahl in combinations:
                pass
            else:
                combinations.append(zahl)

    for k in range(0,M):
        for l in range(k,M):
            zahl = ([k],[l])
            zahl_alt = ([l],[k])
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    for v in range(1,n+1):
        for w in range(v,n+1):
            zahl = (v,w)
            zahl_alt = (w,v)
            if zahl in combinations:
                pass
            else:
                if zahl_alt in combinations:
                    pass
                else:
                    combinations.append(zahl)

    print("Nach Nebenbedingung #3")
    print(combinations)
    print(len(combinations))
    print("-----------")

    for u in range(1,n+m-1):
        for v in range(1,n+m-1):
            for j in range(1,m):
                zahl = ((u,j),(v,j+1))
                zahl_alt = ((v,j+1),(u,j))

                zahl2 = ((u,j),(v-1,j+1))
                zahl2_alt = ((v-1,j+1),(u,j))

                zahl3 = ((u-1,j),(v,j+1))
                zahl3_alt = ((v,j+1),(u-1,j))

                zahl4 = ((u-1,j),(v-1,j+1))
                zahl4_alt = ((v-1,j+1),(u-1,j))

                if zahl in combinations:
                    pass
                else:
                    if zahl_alt in combinations:
                        pass
                    else:
                        if u>n and v>n:
                            pass
                        else:
                            if u == n+m-2:
                                pass
                            else:
                                if v == n+m-2:
                                    pass
                                else:
                                    combinations.append(zahl)

                if zahl2 in combinations:
                    pass
                else:
                    if zahl2_alt in combinations:
                        pass
                    else:
                        if v-1 == 0:
                            pass
                        else:
                            if u>n and v-1>n:
                                pass
                            else:
                                if u == n + m - 2:
                                    pass
                                else:
                                    combinations.append(zahl2)

                if zahl3 in combinations:
                    pass
                else:
                    if zahl3_alt in combinations:
                        pass
                    else:
                        if u-1 ==0:
                            pass
                        else:
                            if u-1>n and v>n:
                                pass
                            else:
                                if v == n + m - 2:
                                    pass
                                else:
                                    combinations.append(zahl3)

                if zahl4 in combinations:
                    pass
                else:
                    if zahl4_alt in combinations:
                        pass
                    else:
                        if u-1 == 0:
                            pass
                        else:
                            if v-1 == 0:
                                pass
                            else:
                                if u-1 >n and v-1>n:
                                    pass
                                else:
                                    combinations.append(zahl4)
    print("Nach Nebenbedingung #4")
    print(combinations)
    print(len(combinations))
    print("-----------")
    return len(combinations)
# def original_model(n,m,M):
#
#     combinations=[]
#
#     #1 constraint
#     for j in range(1, m+1):
#         for v in range(1,n+m-1):
#             for u in range(v+1,n+m-1):
#                 zahl = ((v,j),(u,j))
#                 zahl_alt = ((u,j),(v,j))
#                 if zahl in combinations:
#                     pass
#                 else:
#                     if zahl_alt in combinations:
#                         pass
#                     else:
#                         combinations.append(zahl)
#
#     #2 constraint
#     for v in range(1,n+1):
#         zahl = (v,v)
#         if zahl in combinations:
#             pass
#         else:
#             combinations.append(zahl)
#         for j in range(1,m+1):
#             azahl = ((v,j), (v,j))
#             if azahl in combinations:
#                 pass
#             else:
#                 combinations.append(azahl)
#
#     for v in range(1,n+1):
#         for j in range(1,m+1):
#             zahl = (v,(v,j))
#             zahl_alt = ((v,j),v)
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#
#     for v in range(1,n+1):
#         for j in range(1,m+1):
#             for k in range(j,m+1):
#                 zahl = ((v,j),(v,k))
#                 zahl_alt = ((v,k),(v,j))
#                 if zahl in combinations:
#                     pass
#                 else:
#                     if zahl_alt in combinations:
#                         pass
#                     else:
#                         combinations.append(zahl)
#
#     for k in range(0,M):
#         combinations.append(([k],[k]))
#         for v in range(1,n+1):
#             zahl = ([k],v)
#             if zahl in combinations:
#                 pass
#             else:
#                 combinations.append(zahl)
#
#     for k in range(0,M):
#         for l in range(k,M):
#             zahl = ([k],[l])
#             zahl_alt = ([l],[k])
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#
#     for v in range(1,n+1):
#         for w in range(v,n+1):
#             zahl = (v,w)
#             zahl_alt = (w,v)
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#
#     for u in range(1,n+m-1):
#         for v in range(1,n+m-1):
#             for j in range(1,m):
#                 zahl = ((u,j),(v,j+1))
#                 zahl_alt = ((v,j+1),(u,j))
#                 if u==v:
#                     pass
#                 elif u > n and v > n:
#                     pass
#                 elif zahl in combinations:
#                     pass
#                 else:
#                     if zahl_alt in combinations:
#                         pass
#                     else:
#                         combinations.append(zahl)
#
#     return len(combinations)
#
# ######################################################################
#
# def domain_wall(n,m,M):
#
#     combinations=[]
#
#     #1 constraint
#     for j in range(1, m+1):
#         for v in range(1,n+m-3):
#             zahl = ((v,j),(v+1,j))
#             zahl_alt = ((v+1,j),(v,j))
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#         einzel = ((1,j),(1,j))
#         einzel2 = ((n+m-3,j),(n+m-3,j))
#         combinations.append(einzel)
#         combinations.append(einzel2)
#
#     #2 constraint
#     for v in range(1,n+1):
#         zahl = (v,v)
#         if zahl in combinations:
#             pass
#         else:
#             combinations.append(zahl)
#         for j in range(1,m+1):
#             azahl = ((v,j), (v,j))
#             if azahl in combinations:
#                 pass
#             else:
#                 combinations.append(azahl)
#
#     for v in range(1,n+1):
#         for j in range(1,m+1):
#             zahl = (v,(v,j))
#             zahl_alt = ((v,j),v)
#             zahl2 = (v,(v-1,j))
#             zahl2_alt = ((v-1,j),v)
#
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#             if zahl2 in combinations:
#                 pass
#             else:
#                 if zahl2_alt in combinations:
#                     pass
#                 else:
#                     if v-1 == 0:
#                         pass
#                     else:
#                         combinations.append(zahl2)
#
#
#     for v in range(1,n+1):
#         for j in range(1,m+1):
#             for k in range(j,m+1):
#                 zahl = ((v,j),(v,k))
#                 zahl_alt = ((v,k),(v,j))
#
#                 zahl2 = ((v,j), (v-1,k))
#                 zahl2_alt = ((v-1,k),(v,j))
#                 if zahl in combinations:
#                     pass
#                 else:
#                     if zahl_alt in combinations:
#                         pass
#                     else:
#                         combinations.append(zahl)
#                 if zahl2 in combinations:
#                     pass
#                 else:
#                     if zahl2_alt in combinations:
#                         pass
#                     else:
#                         if v-1 == 0:
#                             pass
#                         else:
#                             combinations.append(zahl2)
#
#     for k in range(0,M):
#         combinations.append(([k],[k]))
#         for v in range(1,n+1):
#             zahl = ([k],v)
#             if zahl in combinations:
#                 pass
#             else:
#                 combinations.append(zahl)
#
#     for k in range(0,M):
#         for l in range(k,M):
#             zahl = ([k],[l])
#             zahl_alt = ([l],[k])
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#
#     for v in range(1,n+1):
#         for w in range(v,n+1):
#             zahl = (v,w)
#             zahl_alt = (w,v)
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#
#     for u in range(1,n+m-1):
#         for v in range(1,n+m-1):
#             for j in range(1,m):
#                 zahl = ((u,j),(v,j+1))
#                 zahl_alt = ((v,j+1),(u,j))
#
#                 zahl2 = ((u,j),(v-1,j+1))
#                 zahl2_alt = ((v-1,j+1),(u,j))
#
#                 zahl3 = ((u-1,j),(v,j+1))
#                 zahl3_alt = ((v,j+1),(u-1,j))
#
#                 zahl4 = ((u-1,j),(v-1,j+1))
#                 zahl4_alt = ((v-1,j+1),(u-1,j))
#
#                 if zahl in combinations:
#                     pass
#                 else:
#                     if zahl_alt in combinations:
#                         pass
#                     else:
#                         if u>n and v>n:
#                             pass
#                         else:
#                             if u == n+m-2:
#                                 pass
#                             else:
#                                 if v == n+m-2:
#                                     pass
#                                 else:
#                                     combinations.append(zahl)
#
#                 if zahl2 in combinations:
#                     pass
#                 else:
#                     if zahl2_alt in combinations:
#                         pass
#                     else:
#                         if v-1 == 0:
#                             pass
#                         else:
#                             if u>n and v-1>n:
#                                 pass
#                             else:
#                                 if u == n + m - 2:
#                                     pass
#                                 else:
#                                     combinations.append(zahl2)
#
#                 if zahl3 in combinations:
#                     pass
#                 else:
#                     if zahl3_alt in combinations:
#                         pass
#                     else:
#                         if u-1 ==0:
#                             pass
#                         else:
#                             if u-1>n and v>n:
#                                 pass
#                             else:
#                                 if v == n + m - 2:
#                                     pass
#                                 else:
#                                     combinations.append(zahl3)
#
#                 if zahl4 in combinations:
#                     pass
#                 else:
#                     if zahl4_alt in combinations:
#                         pass
#                     else:
#                         if u-1 == 0:
#                             pass
#                         else:
#                             if v-1 == 0:
#                                 pass
#                             else:
#                                 if u-1 >n and v-1>n:
#                                     pass
#                                 else:
#                                     combinations.append(zahl4)
#     return len(combinations)
#
# def reducedwithdomain_y(n,m,M):
#
#     combinations=[]
#
#     #1 constraint
#     for j in range(1, m+1):
#         for v in range(1,n+m-3):
#             zahl = ((v,j),(v+1,j))
#             zahl_alt = ((v+1,j),(v,j))
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#         einzel = ((1,j),(1,j))
#         einzel2 = ((n+m-3,j),(n+m-3,j))
#         combinations.append(einzel)
#         combinations.append(einzel2)
#
#     #2 constraint
#     for v in range(1,n+1):
#         for j in range(1,m+1):
#             for i in range(j,m+1):
#                 zahl = ((v,j), (v,i))
#                 zahl_alt= ((v,i), (v,j))
#
#                 zahl2 = ((v,j), (v-1,i))
#                 zahl2_alt= ((v,i), (v,j))
#
#                 zahl3 = ((v-1,j), (v,i))
#                 zahl3_alt= ((v,i), (v-1,j))
#
#                 zahl4 = ((v-1,j), (v-1,i))
#                 zahl4_alt= ((v-1,i), (v-1,j))
#
#                 if zahl in combinations:
#                     pass
#                 else:
#                     if zahl_alt in combinations:
#                         pass
#                     else:
#                         combinations.append(zahl)
#
#                 if v-1 == 0:
#                     zahl2 = ((v,i),(v,i))
#                     zahl2_alt = ((v,i),(v,i))
#
#                     zahl3 = ((v, j), (v, j))
#                     zahl3_alt = ((v, j), (v, j))
#                 if zahl2 in combinations:
#                     pass
#                 else:
#                     if zahl2_alt in combinations:
#                         pass
#                     else:
#                         combinations.append(zahl2)
#
#                 if zahl3 in combinations:
#                     pass
#                 else:
#                     if zahl3_alt in combinations:
#                         pass
#                     else:
#                         combinations.append(zahl3)
#
#
#     for k in range(0,M):
#         combinations.append(([k],[k]))
#         for v in range(1,n+1):
#             zahl = ([k],v)
#             if zahl in combinations:
#                 pass
#             else:
#                 combinations.append(zahl)
#
#     for k in range(0,M):
#         for l in range(k,M):
#             zahl = ([k],[l])
#             zahl_alt = ([l],[k])
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#
#     for v in range(1,n+1):
#         for w in range(v,n+1):
#             zahl = (v,w)
#             zahl_alt = (w,v)
#             if zahl in combinations:
#                 pass
#             else:
#                 if zahl_alt in combinations:
#                     pass
#                 else:
#                     combinations.append(zahl)
#
#     for u in range(1,n+m-1):
#         for v in range(1,n+m-1):
#             for j in range(1,m):
#                 zahl = ((u,j),(v,j+1))
#                 zahl_alt = ((v,j+1),(u,j))
#
#                 zahl2 = ((u,j),(v-1,j+1))
#                 zahl2_alt = ((v-1,j+1),(u,j))
#
#                 zahl3 = ((u-1,j),(v,j+1))
#                 zahl3_alt = ((v,j+1),(u-1,j))
#
#                 zahl4 = ((u-1,j),(v-1,j+1))
#                 zahl4_alt = ((v-1,j+1),(u-1,j))
#
#                 if zahl in combinations:
#                     pass
#                 else:
#                     if zahl_alt in combinations:
#                         pass
#                     else:
#                         if u>n and v>n:
#                             pass
#                         else:
#                             if u == n+m-2:
#                                 pass
#                             else:
#                                 if v == n+m-2:
#                                     pass
#                                 else:
#                                     combinations.append(zahl)
#
#                 if zahl2 in combinations:
#                     pass
#                 else:
#                     if zahl2_alt in combinations:
#                         pass
#                     else:
#                         if v-1 == 0:
#                             pass
#                         else:
#                             if u>n and v-1>n:
#                                 pass
#                             else:
#                                 if u == n + m - 2:
#                                     pass
#                                 else:
#                                     combinations.append(zahl2)
#
#                 if zahl3 in combinations:
#                     pass
#                 else:
#                     if zahl3_alt in combinations:
#                         pass
#                     else:
#                         if u-1 ==0:
#                             pass
#                         else:
#                             if u-1>n and v>n:
#                                 pass
#                             else:
#                                 if v == n + m - 2:
#                                     pass
#                                 else:
#                                     combinations.append(zahl3)
#
#                 if zahl4 in combinations:
#                     pass
#                 else:
#                     if zahl4_alt in combinations:
#                         pass
#                     else:
#                         if u-1 == 0:
#                             pass
#                         else:
#                             if v-1 == 0:
#                                 pass
#                             else:
#                                 if u-1 >n and v-1>n:
#                                     pass
#                                 else:
#                                     combinations.append(zahl4)
#     return len(combinations)
# failure= []
# quo = []
# for n in range(2,20):
#     print("wir sind bei", n)
#     for m in range(2,n):
#         for M in range(2,10):
#             difference = original_model(n,m,M)-domain_wall(n,m,M)
#             quotient = original_model(n,m,M)/domain_wall(n,m,M)
#             if difference < 0:
#                 failure.append((n,m,M))
#             quo.append(quotient)
#             savings = sum(quo)/len(quo)
#
#
# print(failure)
# print(savings)

# original = []
# domain = []
# reducing = []
# combi = []
# for n in range(2,11):
#     for m in range(1,n+1):
#         for M in range(1,5):
#             print(n,m,M)
#             o = original_model(n,m,M)
#             d = domain_wall(n,m,M)
#             r = reduced(n,m,M)
#             c = domain_and_reduced_y(n,m,M)
#             domain.append(d/o)
#             reducing.append(r/o)
#             combi.append(c/o)
#
# print("Avg ratio domain/original:", sum(domain)/len(domain))
# print("Avg ratio reduced/original:", sum(reducing)/len(reducing))
# print("Avg ratio combi/original:", sum(combi)/len(combi))
def test_plot():
    list=[]
    o = []
    d = []
    m = []
    dr = []
    r = []
    tsp = []
    for t in range(2,16):
        list.append(t)
        tsp.append(int(t**2+2*t**2*(t-1)))
        o.append(original_model(t,t,4))
        d.append(domain_wall(t,t,4))
        m.append(more_variables(t,t,4))
        #dr.append(domain_and_reduced_y(t,t,4))
        #r.append(reduced(t,t,4))
    plt.plot(list, o, 'r--', list, d, 'g--', list, m, 'y--', list, tsp, 'c--')
    plt.show()

def test_plot2():
    list=[]
    o = []
    d = []
    m = []
    dr = []
    r = []
    for t in range(2,16):
        list.append(t)
        o.append(original_model(15,t,4))
        d.append(domain_wall(15,t,4))
        m.append(more_variables(15,t,4))
        dr.append(domain_and_reduced_y(15,t,4))
        r.append(reduced(15,t,4))
    plt.plot(list, o, 'r--', list, d, 'g--', list, m, 'y--', list, dr, 'b--', list, r, '--')
    plt.show()