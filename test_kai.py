### implementation CPCTSP with different models###
import dimod
import networkx as nx
import gurobipy as gp
import numpy as np
import cvrplib
import matplotlib.pyplot as plt
import itertools
import time as tm
import datetime as dt
from dimod import BinaryQuadraticModel
import dimod as dm
import matplotlib
from greedy import SteepestDescentSolver
from dwave.cloud import Client
from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite, DWaveCliqueSampler, LeapHybridSampler
import json
import tsplib95

##def log trick

def main():
    '''run this function to obtain a solution for the instance in the input,
    viewed as Capacitated Price-Collecting Traveling Salesman Problem'''

    ### Load and process instance
    instance = tsplib95.load('C:/Users/kaiha/Desktop/ocean/Samples/eil13.vrp')

    n = instance.dimension - 1
    C = instance.capacity

    demands = list(instance.demands.values())
    demandssorted = list(instance.demands.values())
    demandssorted.pop(0) ## in the instance, the depot also got a demand (which is 0). This has to be eliminated
    gcd = np.gcd.reduce(demandssorted+[C])
    demandssorted.sort() #ascending
    dmin = demandssorted[0]/gcd
    load = demandssorted.pop(0)
    m = 0
    while load <= C:
        m += 1
        if demandssorted:
            load += demandssorted.pop(0)
        else:
            break
    C = C/gcd
    M = int(np.ceil(np.log2(C - dmin + 1)))

    H = nx.complete_graph(n+1)
    #{1: 20.0, 2: 40.0, 3: 50.0, 4: 50.0, 5: 40.0, 6: 20.0} prices for eil7

    #prices = [[0,20,40,50,50,40,20]]
    prices = 30*np.random.rand(1,n+1)
    prices[0][0] = 0
    for e in list(H.edges()):
        H[e[0]][e[1]]['length'] = instance.get_weight(*(e[0]+1,e[1]+1))
    for v in H.nodes():
        H.nodes[v]['price'] = prices[0][v]
        H.nodes[v]['demand'] = demands[v]/gcd

    for i in range(1,m - 1):
        H.add_edges_from([(n + i, v, {'length': H[0][v]['length']}) for v in H.nodes() if v != 0])
        H.add_edge(n + i, 0, length=0)
        H.nodes[n + i]['demand'] = 0
        H.nodes[n + i]['price'] = 0


    pricessorted = [H.nodes[v]['price'] for v in range(1, n+1)]
    pricessorted.sort(reverse=True)

    HD = nx.DiGraph(H)

    B = 1
    A = (m+1)*max(H[u][v]['length'] for u,v in H.edges) + sum(pricessorted[v] for v in range(m))

    ###print parameters for checks
    print("n=",n)
    print("C=",C)
    print("dmin=",dmin)
    print("m=", m)
    print("A=",A)
    print("M=",M)

    ### Find solutions to the problem, decide which model should be used
    #solution = domain_wall(HD, n, m, C, M, A, B, dmin)
    #solution = domain_wall_updated(HD, n, m, C, M, A, B, dmin)
    #solution = combination(HD, n, m, C, M, A, B, dmin)
    solution = perfect_solution(HD, n, m, C, M, A, B, dmin)



    #for key,value in solution.sample.items():
    #    print(f"{key}: {value}")

    #drawing(n, solution, m)
    drawing_gurobi(n, solution, m)

    return solution


def domain_wall(HD, n, m, C, M, A, B, dmin):
    '''This function will run the sample through the model using domain wall encoding as further enhancement
    of the previous model. The variables are binary.'''


    ### variables
    s = {}
    y = {}
    w = {}

    for j in range(1,m+1):
        for v in range(1, n + m - 2):
            s[v,j] = BinaryQuadraticModel({f's_{v}_{j}':1},{},0,vartype='BINARY')
        s[0,j] = 0
        s[n+m-2,j] = 1

    for v in range(1,n+1):
        y[v] = BinaryQuadraticModel({f'y_{v}':1},{},0,vartype='BINARY')

    for k in range(M):
        w[k] = BinaryQuadraticModel({f'w_{k}': 1}, {}, 0, vartype='BINARY')

    ### building the QUBO. HA insures the feasibility, HB the optimality

    HA = (A*dm.quicksum( (n+m-4) - dm.quicksum( (2*s[v,j] - 1)*(2*s[v+1,j] - 1) for v in range(n+m-2)) for j in range(1,m+1)) +
          A*dm.quicksum(y[v] - dm.quicksum( s[v,j] - s[v-1,j] for j in range(1,m+1)) for v in range(1, n+1))**2 +
          A*( dmin + dm.quicksum( 2**k * w[k] for k in range(M-1)) + (C - dmin - 2**(M-1) + 1)*w[M-1] - dm.quicksum(HD.nodes[v]['demand']*y[v] for v in range(1,n+1)))**2)

    HB = (B*dm.quicksum((HD[u][v]['length']-HD.nodes[u]['price']/2-HD.nodes[v]['price']/2)*dm.quicksum((s[u,j] - s[u-1,j])*(s[v,j+1] - s[v-1,j+1]) for j in range(1,m)) for u in range(1,m+n-1) for v in range(1,m+n-1) if u!=v) +
          B*dm.quicksum((HD[0][v]['length']-HD.nodes[v]['price']/2)*(s[v,1] - s[v-1,1]) for v in range(1,n+1))**2 +
          B*dm.quicksum((HD[0][v]['length']-HD.nodes[v]['price']/2)*(s[v,m] - s[v-1,m]) for v in range(1,n+1))**2)

    qubo = HA + HB
    qubo = BinaryQuadraticModel(qubo.linear, {interaction: bias for interaction, bias in qubo.quadratic.items() if bias}, qubo.offset, qubo.vartype())

    sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver={'qpu': True}, profile = 'prod'))
    response = sampler.sample(qubo, num_reads=1000, annealing_time = 5)

    #solver_greedy = SteepestDescentSolver()
    #response = solver_greedy.sample(qubo, initial_states=response)

    ### find the solution with the lowest energy and is feasible

    #min_energy_sample = response.first

    for solution in response.data(['sample', 'energy']):
        if isfeasible(solution, A, dmin, n, m, M, C, HD,0):
            min_energy_sample = solution
            break
        else:
            continue

    return min_energy_sample

def isfeasible(data, A, dmin, n, m, M, C, HD, version):
    '''This function checks the feasibility of the found solution'''
    t=0
    if m>4:
        t=1

    for j in range(1,m+1):
        data.sample[f's_0_{j}'] = 0
        data.sample[f's_{n+1+t}_{j}'] = 1
    HA0 =0
    if version == 1:
        HA0=  A*sum(sum((data.sample[f'x_{v}_{j}'] - data.sample[f's_{v}_{j}'] + data.sample[f's_{v-1}_{j}'])**2 for j in range(1,m+1)) for v in range(1,n+2+t))
    HA1 = A*sum( (n-1) - sum( (2*data.sample[f's_{v}_{j}'] - 1)*(2*data.sample[f's_{v+1}_{j}'] - 1) for v in range(n+1+t)) for j in range(1,m+1))
    HA2=  A*sum(((data.sample[f'y_{v}'] - sum( data.sample[f's_{v}_{j}'] - data.sample[f's_{v-1}_{j}'] for j in range(1,m+1)))**2) for v in range(1, n+1))
    HA3=  A*( dmin + sum( 2**k * data.sample[f'w_{k}'] for k in range(M-1)) + (C - dmin - 2**(M-1) + 1)*data.sample[f'w_{M-1}'] - sum(HD.nodes[v]['demand']*data.sample[f'y_{v}'] for v in range(1,n+1)))**2

    HA = HA0+HA1+HA2+HA3
    if HA == 0:
        feasible = True
        print(HA0, HA1, HA2, HA3)
    else:
        feasible = False
    return feasible

def drawing(n, solution, m):
    '''This function draws the solution on a plot'''
    ### Using the choosed solution and display it on a graph
    t=0
    if m>4:
        t=1

    G = nx.DiGraph()
    G.add_nodes_from(range(n+2+t))

    for i in range(1, n+2+t):
        for k in range(1,n+2+t):
            if i !=k:
                for j in range(1, m):
                    if solution.sample[f"s_{i}_{j}"] - solution.sample[f"s_{i - 1}_{j}"] == 1 and solution.sample[f"s_{k}_{j + 1}"] - solution.sample[f"s_{k-1}_{j + 1}"] == 1:
                        G.add_edges_from([(i, k)])

    for i in range(1, n+2+t):
        if solution.sample[f"s_{i}_{1}"] - solution.sample[f"s_{i - 1}_{1}"] == 1:
            G.add_edges_from([(0, i)])
        if solution.sample[f"s_{i}_{m}"] - solution.sample[f"s_{i - 1}_{m}"] == 1:
            G.add_edges_from([(i, 0)])
    labels = {i: i for i in range(n+2+t)}

    def get_node_color(node):
        if node == 0:
            return 'lightblue'
        elif node > n:
            return 'gray'
        else:
            return 'lightgray'

    node_colors = [get_node_color(node) for node in G.nodes()]


    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.text(0.5, 1.05, f"{n} nodes, max route length {m}, energy {solution.energy}", horizontalalignment='center', transform=plt.gca().transAxes,
             fontsize=10)
    plt.axis('off')
    plt.show()

    return 0

def drawing_gurobi(n, solution, m):
    '''This function draws the solution on a plot'''
    ### Using the choosed solution and display it on a graph
    t=0
    if m>4:
        t=1

    G = nx.DiGraph()
    G.add_nodes_from(range(n+2+t))

    for i in range(1, n+2+t):
        for k in range(1,n+2+t):
            if i !=k:
                for j in range(1, m):
                    if solution[f"s_{i}_{j}"] - solution[f"s_{i - 1}_{j}"] == 1 and solution[f"s_{k}_{j + 1}"] - solution[f"s_{k-1}_{j + 1}"] == 1:
                        G.add_edges_from([(i, k)])

    for i in range(1, n+2+t):
        if solution[f"s_{i}_{1}"] - solution[f"s_{i - 1}_{1}"] == 1:
            G.add_edges_from([(0, i)])
        if solution[f"s_{i}_{m}"] - solution[f"s_{i - 1}_{m}"] == 1:
            G.add_edges_from([(i, 0)])
    labels = {i: i for i in range(n+2+t)}

    def get_node_color(node):
        if node == 0:
            return 'lightblue'
        elif node > n:
            return 'gray'
        else:
            return 'lightgray'

    node_colors = [get_node_color(node) for node in G.nodes()]


    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.text(0.5, 1.05, f"{n} nodes, max route length {m}", horizontalalignment='center', transform=plt.gca().transAxes,
             fontsize=10)
    plt.axis('off')
    plt.show()

    return 0

def domain_wall_updated(HD, n, m, C, M, A, B, dmin):
    '''This function will run the sample through the model using domain wall encoding as further enhancement
    of the previous model. The variables are binary. UPDATE: There is just one/two dummy depot!'''


    ### variables
    s = {}
    y = {}
    w = {}

    t=0
    if m>4:
        t=1

    for j in range(1,m+1):
        for v in range(1, n + t + 1):
            s[v,j] = BinaryQuadraticModel({f's_{v}_{j}':1},{},0,vartype='BINARY')
        s[0,j] = 0
        s[n+1 +t,j] = 1

    for v in range(1,n+1):
        y[v] = BinaryQuadraticModel({f'y_{v}':1},{},0,vartype='BINARY')

    for k in range(M):
        w[k] = BinaryQuadraticModel({f'w_{k}': 1}, {}, 0, vartype='BINARY')

    ### building the QUBO. HA insures the feasibility, HB the optimality

    HA = (A*dm.quicksum( (n-1+t) - dm.quicksum( (2*s[v,j] - 1)*(2*s[v+1,j] - 1) for v in range(n+1+t)) for j in range(1,m+1)) +
          A*dm.quicksum(y[v] - dm.quicksum( s[v,j] - s[v-1,j] for j in range(1,m+1)) for v in range(1, n+1))**2 +
          A*( dmin + dm.quicksum( 2**k * w[k] for k in range(M-1)) + (C - dmin - 2**(M-1) + 1)*w[M-1] - dm.quicksum(HD.nodes[v]['demand']*y[v] for v in range(1,n+1)))**2)

    HB = (B*dm.quicksum((HD[u][v]['length']-HD.nodes[u]['price']/2-HD.nodes[v]['price']/2)*dm.quicksum((s[u,j] - s[u-1,j])*(s[v,j+1] - s[v-1,j+1]) for j in range(1,m)) for u in range(1,n+2+t) for v in range(1,n+2+t) if u!=v) +
          B*dm.quicksum((HD[0][v]['length']-HD.nodes[v]['price']/2)*(s[v,1] - s[v-1,1]) for v in range(1,n+1))**2 +
          B*dm.quicksum((HD[0][v]['length']-HD.nodes[v]['price']/2)*(s[v,m] - s[v-1,m]) for v in range(1,n+1))**2)

    qubo = HA + HB
    qubo = BinaryQuadraticModel(qubo.linear, {interaction: bias for interaction, bias in qubo.quadratic.items() if bias}, qubo.offset, qubo.vartype())

    sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver={'qpu': True}, profile = 'prod'))
    response = sampler.sample(qubo, num_reads=1000, annealing_time = 5)

    #solver_greedy = SteepestDescentSolver()
    #response = solver_greedy.sample(qubo, initial_states=response)

    ### find the solution with the lowest energy and is feasible


    min_energy_sample = response.first
    for j in range(1,m+1):
        min_energy_sample.sample[f's_0_{j}'] = 0
        min_energy_sample.sample[f's_{n+1+t}_{j}'] = 1
    for solution in response.data(['sample', 'energy']):
        if isfeasible(solution, A, dmin, n, m, M, C, HD,0):
            min_energy_sample = solution
            break
        else:
            continue

    return min_energy_sample

def combination(HD, n, m, C, M, A, B, dmin):
    '''This function will run the sample through the model using domain wall encoding as further enhancement
    of the previous model. The variables are binary. UPDATE: There is just one/two dummy depot!'''


    ### variables
    x = {}
    s = {}
    y = {}
    w = {}

    t=0
    if m>4:
        t=1

    for j in range(1,m+1):
        for v in range(1, n+t+2):
            x[v,j] = BinaryQuadraticModel({f'x_{v}_{j}':1},{},0,vartype='BINARY')

    for j in range(1,m+1):
        for v in range(1, n + t + 1):
            s[v,j] = BinaryQuadraticModel({f's_{v}_{j}':1},{},0,vartype='BINARY')
        s[0,j] = 0
        s[n+1 +t,j] = 1

    for v in range(1,n+1):
        y[v] = BinaryQuadraticModel({f'y_{v}':1},{},0,vartype='BINARY')

    for k in range(M):
        w[k] = BinaryQuadraticModel({f'w_{k}': 1}, {}, 0, vartype='BINARY')

    ### building the QUBO. HA ensures the feasibility, HB the optimality

    HA = (A*dm.quicksum(dm.quicksum(x[v,j] - s[v,j] + s[v-1,j] for j in range(1,m+1))**2 for v in range(1,n+2+t)) +
          A*dm.quicksum( (n-1+t) - dm.quicksum( (2*s[v,j] - 1)*(2*s[v+1,j] - 1) for v in range(n+1+t)) for j in range(1,m+1)) +
          A*dm.quicksum(y[v] - dm.quicksum( x[v,j] for j in range(1,m+1)) for v in range(1, n+1))**2 +
          A*( dmin + dm.quicksum( 2**k * w[k] for k in range(M-1)) + (C - dmin - 2**(M-1) + 1)*w[M-1] - dm.quicksum(HD.nodes[v]['demand']*y[v] for v in range(1,n+1)))**2)

    HB = (B*dm.quicksum((HD[u][v]['length']-HD.nodes[u]['price']/2-HD.nodes[v]['price']/2)*dm.quicksum((s[u,j] - s[u-1,j])*(s[v,j+1] - s[v-1,j+1]) for j in range(1,m)) for u in range(1,n+2+t) for v in range(1,n+2+t) if u!=v) +
          B*dm.quicksum((HD[0][v]['length']-HD.nodes[v]['price']/2)*(s[v,1] - s[v-1,1]) for v in range(1,n+1))**2 +
          B*dm.quicksum((HD[0][v]['length']-HD.nodes[v]['price']/2)*(s[v,m] - s[v-1,m]) for v in range(1,n+1))**2)

    qubo = HA + HB
    qubo = BinaryQuadraticModel(qubo.linear, {interaction: bias for interaction, bias in qubo.quadratic.items() if bias}, qubo.offset, qubo.vartype())

    sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver={'qpu': True}, profile = 'prod'))
    response = sampler.sample(qubo, num_reads=1000, annealing_time = 5)

    solver_greedy = SteepestDescentSolver()
    response = solver_greedy.sample(qubo, initial_states=response)

    ### find the solution with the lowest energy and is feasible


    min_energy_sample = response.first
    for j in range(1,m+1):
        min_energy_sample.sample[f's_0_{j}'] = 0
        min_energy_sample.sample[f's_{n+1+t}_{j}'] = 1
    for solution in response.data(['sample', 'energy']):
        if isfeasible(solution, A, dmin, n, m, M, C, HD, version=1):
            min_energy_sample = solution
            break
        else:
            continue

    return min_energy_sample

def perfect_solution(HD, n, m, C, M, A, B, dmin):
    '''This method won't use Quantum Annealing to solve the problem. Instead, it tries to solve the problem with the package gurobipy.
    We implement the model of domain_wall_updated.'''

    model = gp.Model()

    ### variables
    s = {}
    y = {}
    w = {}

    t=0
    if m>4:
        t=1

    for j in range(1,m+1):
        for v in range(1, n + t + 1):
            s[v,j] = model.addVar(name=f's_{v}_{j}', vtype='b')
        s[0,j] = 0
        s[n+1 +t,j] = 1

    for v in range(1,n+1):
        y[v] = model.addVar(name=f'y_{v}', vtype='b')

    for k in range(M):
        w[k] = model.addVar(name=f'w_{k}', vtype='b')

    ### building the QUBO.
    model.update()
    model.setObjective(
            A*gp.quicksum( (n-1+t) - gp.quicksum( (2*s[v,j] - 1)*(2*s[v+1,j] - 1) for v in range(n+1+t)) for j in range(1,m+1)) +\
            A*gp.quicksum((y[v] - gp.quicksum( s[v,j] - s[v-1,j] for j in range(1,m+1)))**2 for v in range(1, n+1)) +\
            A*( dmin + gp.quicksum( 2**k * w[k] for k in range(M-1)) + (C - dmin - 2**(M-1) + 1)*w[M-1] - gp.quicksum(HD.nodes[v]['demand']*y[v] for v in range(1,n+1)))**2+\
            B*gp.quicksum((HD[u][v]['length']-HD.nodes[u]['price']/2-HD.nodes[v]['price']/2)*gp.quicksum((s[u,j] - s[u-1,j])*(s[v,j+1] - s[v-1,j+1]) for j in range(1,m)) for u in range(1,n+2+t) for v in range(1,n+2+t) if u!=v) +\
            B*gp.quicksum(((HD[0][v]['length']-HD.nodes[v]['price']/2)*(s[v,1] - s[v-1,1]))**2 for v in range(1,n+1)) + \
            B*gp.quicksum(((HD[0][v]['length']-HD.nodes[v]['price']/2)*(s[v,m] - s[v-1,m]))**2 for v in range(1,n+1)))

    model.optimize()

    solution = {}
    for v in model.getVars():
        solution[v.varName] = v.x

    for j in range(1,m+1):
        solution[f's_0_{j}'] = 0
        solution[f's_{n+1+t}_{j}'] = 1

    print("Objective value:", model.ObjVal)

    return solution

def pricing():
    '''This function can calculate the prices for every given instance, in order to insert them into main()'''
    instance = tsplib95.load('C:/Users/kaiha/Desktop/ocean/Samples/eil7.vrp')
    n = instance.dimension - 1
    C = instance.capacity

    demands = list(instance.demands.values())
    H = nx.complete_graph(n + 1)

    for e in list(H.edges()):
        H[e[0]][e[1]]['length'] = instance.get_weight(*(e[0] + 1, e[1] + 1))
    for v in H.nodes():
        H.nodes[v]['demand'] = demands[v]

    model = gp.Model()
    pi = model.addVars(range(1, n+1), name='pi')
    model.setObjective(sum(pi[v] for v in range(1, n)), sense=gp.GRB.MAXIMIZE)
    model.update()

    ShortRoutes = {v: [(0, v), (v, 0)] for v in range(1, n)}
    ShortCosts = {v: 2 * H[0][v]['length'] for v in range(1, n)}
    Routes, Costs = list(ShortRoutes.values()), list(ShortCosts.values())
    for k, r in enumerate(Routes):
        model.addConstr(sum(pi[v] for v in range(1, n+1) if [e for e in r if v in e] != []) <= Costs[k])
    model.optimize()

    price = {}
    for v in model.getVars():
        price[v.varName] = v.x

    return price

#model = main()
#print(model)

price = pricing()
print(price)