from random import randint
import random as rn
import numpy as np
import collections
import networkx as nx
import sys
import os
import csv
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import sys
import getopt
from os import path

def compute_I(d):
    d_i = d[0]
    res = 0
    for d_j in d:
        res += d_i - d_j
    return res


def c_merge(d, d1, k):
    res = d1 - d[k] + compute_I(d[k+1: min(len(d), 2*k)])
    return res


def c_new(d, k):
    t = d[k:min(len(d), 2*k-1)]
    res = compute_I(t)
    return res


def greedy_rec_algorithm(array_degrees, k_degree, pos_init, extension):
    # complete this function
    if pos_init + extension >= len(array_degrees) - 1:
        for i in range(pos_init, len(array_degrees)):
            array_degrees[i] = array_degrees[pos_init]
        return array_degrees
    else:
        d1 = array_degrees[pos_init]
        c_merge_cost = c_merge(array_degrees, d1, pos_init + extension)
        c_new_cost = c_new(d, pos_init + extension)

        if c_merge_cost > c_new_cost:
            for i in range(pos_init, pos_init + extension):
                array_degrees[i] = d1
            greedy_rec_algorithm(array_degrees, k_degree,
                                 pos_init + extension, k_degree)
        else:
            greedy_rec_algorithm(array_degrees, k_degree,
                                 pos_init, extension + 1)

    return array_degrees


def compute_I_range(d, start, end):
    d_i = d[0]
    res = 0
    for i in range(start, end+1):
        res += d_i - d[i]
    return res


def dp_algorithm(array_degrees_dp, k):

    tmp_array = array_degrees_dp.copy()

    cost_anonimization = []
    for i in range(0, len(array_degrees_dp)-1):
        # passo base i<2k
        if i < (2*k - 1):
            cost_min2k = compute_I_range(array_degrees_dp, 0, i)
            cost_anonimization.append(cost_min2k)
        # passo induttivo
        else:
            ind_min = (max(k, i - 2*k + 1)) - 1
            ind_max = i - k
            for t in range(ind_min, ind_max+1):

                cost_ind_min = compute_I_range(
                    array_degrees_dp, 0, ind_min) + compute_I_range(array_degrees_dp, ind_min+1, i)

                new_cost = compute_I_range(array_degrees_dp, 0, t) + \
                    compute_I_range(array_degrees_dp, t+1, i)
                if new_cost < cost_ind_min:
                    if not new_cost in cost_anonimization:
                        cost_anonimization.append(new_cost)
                else:
                    if not cost_ind_min in cost_anonimization:
                        cost_anonimization.append(cost_ind_min)

    lung = len(tmp_array)
    final_array = []
    for i in range(0, lung-1, k):
        first_element = tmp_array[i]
        # print("f", first_element)
        for j in range(0, k):
            final_array.append(first_element)

    if lung % k != 0:
        mod = lung % k
        for ind in range(0, mod):
            final_array.append(final_array[len(final_array) - 1])

    print("DP algorithm cost", cost_anonimization)
    #print(len(final_array))

    return final_array


def supergraph(originalGraph1, degree_seq_anonymized, degree_seq, array_index):

    originalGraph = originalGraph1.copy()

    a = []
    
    # computo a = degree_seq_anonymized - degree_seq
    for i in range(0, len(degree_seq)-1):
        a.append(degree_seq_anonymized[i]-degree_seq[i])
    
    if ((np.sum(a) % 2) == 1):
        return None

    l = 50
    
    Vl_tmp = np.sort(a)[::-1]

    # prendo gli l elementi più grandi
    Vl = []
    for j in range(0, l):
        Vl.append(Vl_tmp[j])          

    # 1) prima somma
    Vl_sum = sum(Vl)

    # 2) seconda somma
    sec_array = []
    count_2 = 0 
    for sec_ind in range(0, len(Vl)):
        tmp = Vl[sec_ind]
        if originalGraph.has_edge(tmp , Vl[sec_ind]):
            count_2 += 1
        sec_array.append(l - 1 - count_2)
    
    sec_sum = sum(sec_array)    
        

    # 3) terza somma
    third_array = 0
    count_3 = 0
    for third_ind in range(len(array_index) - l):
        tmp = array_index[third_ind] 
        if originalGraph.has_edge(third_ind, Vl[third_ind]):
            count_3 += 1
        third_array += min (l - count_3, a[third_ind])
    
    if (Vl_sum > sec_sum + third_array):
        return None
    

    # sceglie un nodo random da originalGraph.nodes e fa originalGraph.add_edge(random(originalGraph), a[v])
    number_edges = 0
    originalGraph_nodes = originalGraph.nodes()

    while True:
        if all(dindex == 0 for dindex in a):
            if number_edges == sum(a) / 2:
                return originalGraph
            else:
                print("Unknown")
                return None

        v = np.random.choice((np.where(np.array(a) > 0))[0])
        # degree di v
        dv = a[v]
        a[v] = 0 

        for final_ind in range(0, len(originalGraph_nodes)-1):
            if not originalGraph.has_edge(v, a[v]):
                if a[final_ind] > 0 and final_ind != v:
                    originalGraph.add_edge(v, a[v])
                    a[final_ind] -= 1
                    dv -= 1
                    number_edges += 1



def construct_graph(degree_sequence):

    V = []
    E = []

    graph_for_construct = nx.Graph()

    for i_ind in range(0, len(degree_sequence)):
        V.append(i_ind)
    
    if (sum(degree_sequence) % 2) != 0: 
        return None
    
    while True:

        for j_ind in range(0, len(degree_sequence)):
            if degree_sequence[j_ind] < 0:
                return None 

            elif all([vertex == 0 for vertex in degree_sequence]):
                for en in range(0, len(E)-1, 2):
                    graph_for_construct.add_edge(E[en], E[en+1])
                return graph_for_construct
        
            v = np.random.choice((np.where(np.array(degree_sequence) > 0))[0])
            dv = degree_sequence[v]

            degree_sequence[v] = 0
            for w in np.argsort(degree_sequence)[-dv:][::-1]:
                E.append(v)
                E.append(w)
                degree_sequence[w] = degree_sequence[w] - 1





def probing(dv,noise):    
    # increase only the degree of the lowest degree nodes, as suggested in the paper
    n = len(dv)
    for v in range(-noise, 0):
        dv[v] = (np.min([dv[v][0]+1,n-1]),dv[v][1])
    return dv      

def priority(degree_sequence,original_G):

    n = len(degree_sequence)
    # if the sum of the degree sequence is odd, the degree sequence isn't realisable
    if np.sum(degree_sequence) % 2 != 0:
        return None
            
    G = nx.empty_graph(n)
    # transform list of degrees in list of (vertex, degree)
    vd = [(v,d) for v,d in enumerate(degree_sequence)]

    while True:
        
        # sort the list of pairs by degree (second element in the pair)
        vd.sort(key=lambda tup: tup[1], reverse=True)
        # if we ended up with a negative degree, the degree sequence isn't realisable
        if vd[-1][1] < 0:
            return None
                
        tot_degree = 0
        for vertex in vd:
            tot_degree = tot_degree + vertex[1]
        # if all the edges required by the degree sequence have been added, G has been created
        if tot_degree == 0:
            return G
        
        # gather all the vertices that need more edges 
        remaining_vertices = [i for i,vertex in enumerate(vd) if vertex[1] > 0]
        # pick a random one
        idx = remaining_vertices[rn.randrange(len(remaining_vertices))]
        v = vd[idx][0]
        # iterate over all the degree-sorted vertices u such that (u,v) is an edge in the original graph
        for i,u in enumerate(vd):
            # stop when we added v_d edges
            if vd[idx][1] == 0:
                break
            # don't add self-loops
            if u[0] == v:
                continue
            # make sure we're not adding the same edge twice..
            if G.has_edge(u[0],v):
                continue
            # add the edge if this exists also in the original graph
            if original_G.has_edge(v,u[0]) and u[1] > 0:
                G.add_edge(v,u[0])
                # decrease the degree of the connected vertex
                vd[i] = (u[0],u[1] - 1)
                # keep track of how many edges we added
                vd[idx] = (v,vd[idx][1] - 1)
                
        # iterate over all the degree-sorted vertices u such that (u,v) is NOT an edge in the original graph
        for i,u in enumerate(vd):
            # stop when we added v_d edges
            if vd[idx][1] == 0:
                break
            # don't add self-loops
            if u[0] == v:
                continue
            # make sure we're not adding the same edge twice..
            if G.has_edge(v,u[0]):
                continue
            # now add edges that are not in the original graph
            if not original_G.has_edge(v,u[0]):
                G.add_edge(v,u[0])
                # decrease the degree of the connected vertex
                vd[i] = (u[0],u[1] - 1)
                # keep track of how many edges we added
                vd[idx] = (v,vd[idx][1] - 1)



def func_check_priority(degree_seq_priority, array_degrees_priority, priority_graph, k_degree):
    tentativo = 1
    noise = 10


    dv = [(d,v) for v, d in priority_graph.degree()]


    res = priority(degree_seq_priority, priority_graph)

    
    while res is None:
        tentativo = tentativo+1
        print("tentativo number",tentativo)  

        dv = probing(dv,noise)
        #degree_sequence,permutation = sort_dv(dv)
        dv.sort()

        anonymised_sequence = greedy_rec_algorithm(array_degrees_priority, k_degree, 0, k_degree)

        new_anonymised_sequence = [None] * len(degree_seq_priority)

        for i in range(0, len(new_anonymised_sequence)):
            new_anonymised_sequence[i] = degree_seq_priority[i]
        degree_seq_priority = new_anonymised_sequence     

        if not nx.is_valid_degree_sequence_erdos_gallai(degree_seq_priority):
            continue
        res = priority(degree_seq_priority, priority_graph)
        if res is None:
            print("the sequence is valid but the graph construction failed")          

    return res


 
if __name__ == "__main__":


    k = int(sys.argv[1])
    file_graph = sys.argv[2]
    G = nx.Graph()

    if os.path.exists(file_graph):
        # if file ist
        with open(file_graph) as f:
            content = f.readlines()
        # read each line
        content = [x.strip() for x in content]
        for line in content:
            # split name inside each line
            names = line.split(",")
            start_node = names[0]
            if start_node not in G:
                G.add_node(start_node)
            for indNode in range(1, len(names)):
                node_to_add = names[indNode]
                if node_to_add not in G:
                    G.add_node(node_to_add)
                G.add_edge(start_node, node_to_add)

    d = [x[1] for x in G.degree()]
    array_indici = np.argsort(d)[::-1]
    array_degrees = np.sort(d)[::-1]

    array_degrees_greedy = array_degrees.copy()
    array_degrees_dp = array_degrees.copy()
    array_degrees_priority = array_degrees.copy()
    array_degrees_supergraph = array_degrees.copy()

    print("Original degree sequence(not sorted)", d)
    print("Original degree sequence(sorted)", array_degrees_greedy)

    # 1st algorithm = greedy
    degree_sequence_greedy = greedy_rec_algorithm(
        array_degrees_greedy, k, 0, k)
    print("Anonymized degree sequence - Greedy Algorithm",degree_sequence_greedy)

    array_degree_supergraph = degree_sequence_greedy.copy()
    degree_seq_priority = degree_sequence_greedy.copy()
    
    
    # dp
    
    a = dp_algorithm(array_degrees_dp, k)
    print("Anonymized degree sequence - DP algorithm", a)
    
    '''
    degree_sequence_greedy[len(
        degree_sequence_greedy)-1] = degree_sequence_greedy[len(degree_sequence_greedy)-1] + 1

    
 
    '''
    # 2nd algorithm = construct + super graph
    degree_sequence_greedy[len(degree_sequence_greedy)-1] = degree_sequence_greedy[len(degree_sequence_greedy)-1] + 1
    #graph_construct = G.copy()


    graph_dp = construct_graph(degree_sequence_greedy)

    nx.draw(graph_dp,with_labels=True)
    plt.draw()
    plt.show()

    supergrafo = supergraph(G, array_degrees_supergraph, array_degrees, array_indici)

    nx.draw(supergrafo,with_labels=True)
    plt.draw()
    plt.show()    



    
    # priority algorithm

    degree_seq_priority[len(degree_seq_priority)-1] = degree_seq_priority[len(degree_seq_priority)-1] + 1
    priority_graph = G.copy()
 
    
    num_edges_in_G = len(set(G.edges()))

    ris = func_check_priority(degree_seq_priority, array_degrees_priority, G, k)

    nx.draw(ris,with_labels=True)
    plt.draw()
    plt.show()


    num_edges_in_both = len(set(ris.edges()))
    
    print("Num archi grafo originale = " + str(nx.number_of_edges(G)))
    print("Num archi grafo anonimizzato = " + str(nx.number_of_edges(ris)))    
 