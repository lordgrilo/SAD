import numpy as np
import pandas as pd
import os, sys
import random
import networkx as nx

class simplex_vertex:
    '''
    Vertex class for simplicial activity driven model
    '''
    import random

    def __init__(self,vertex_id,activity):
        self.act = activity;
        self.name = vertex_id;
        self.memory = [];
    
    def update_history(self,newcollab):
        self.memory.append(newcollab);
                
    def repeat_collaboration(self):
        s = random.choice(self.memory);
        self.memory.append(s)
        return list(s);

    def new_collab(self,nodes,size):
        el = random.sample(nodes,size-1);
        el.append(self.name);
        self.memory.append(frozenset(el));
        return list(el);

    def new_edge(self,nn):
        self.memory.append(frozenset([self.name,nn]));




def add_clique(e,tgraph):
    g = nx.complete_graph(len(e));
    rl = dict(zip(range(len(e)), e));
    g = nx.relabel_nodes(g,rl)
    tgraph.add_edges_from(g.edges());
    return tgraph;

def add_k_edges(n, e ,tgraph):
    new_neigh = map(lambda x: [n,x], e);
    tgraph.add_edges_from(new_neigh);
    return tgraph;


def memory_instant_graph(vertex_dict,k,mode='simplicial',alpha=1):
    from scipy.stats import norm
    if np.isscalar(k):
        kk = dict.fromkeys(vertex_dict.keys(),k);
    elif len(k)==2:
        dis = norm(loc=k[0],scale=k[1]);
        vv = map(lambda x: int(np.max([2, x])), dis.rvs(size=len(vertex_dict)));     
        kk = dict(zip(vertex_dict.keys(),vv))
    elif len(k)==len(vertex_dict):
        kk = dict(zip(vertex_dict.keys(),k))
        
    tg = nx.Graph();
    tg.add_nodes_from(vertex_dict.keys());
    new_history = [];
    
    if mode=='simplicial':
        for n in vertex_dict:
            if np.random.rand()<=vertex_dict[n].act:
                memory_factor = (1.0 / float(1.0+len(set(vertex_dict[n].memory))))**alpha;
                # memory factor take from 
                #Sun, Kaiyuan, Andrea Baronchelli, and Nicola Perra. 
                #"Epidemic spreading in non-Markovian time-varying networks." (2014).
                if np.random.rand() <= memory_factor:
                    nodes = vertex_dict.keys();
                    nodes.remove(n)
                    e = vertex_dict[n].new_collab(nodes,kk[n]);
                else:
                    e = vertex_dict[n].repeat_collaboration();
                new_history.append(e);
                tg = add_clique(e,tg);
    
    if mode=='network':
        m = {};
        for n in vertex_dict:
            m[n] = kk[n]*(kk[n]-1)/2;
        for n in vertex_dict:
            if np.random.rand()<=vertex_dict[n].act:
                memory_factor = (1.0 / float(1.0+len(set(vertex_dict[n].memory))))**alpha;
                # memory factor take from 
                #Sun, Kaiyuan, Andrea Baronchelli, and Nicola Perra. 
                #"Epidemic spreading in non-Markovian time-varying networks." (2014).
                if np.random.rand() <= memory_factor:
                    nodes = vertex_dict.keys();
                    nodes.remove(n)
                    neigh = random.sample(nodes,m[n]);
                    vertex_dict[n].new_collab(neigh,len(neigh));
                    for nn in neigh:
                        new_history.append([n,nn]);
                    tg = add_k_edges(n,neigh,tg);
                else:
                    neigh =[]
                    e = vertex_dict[n].repeat_collaboration();
                    e.remove(n);
                    for nn in e:
                        new_history.append([n,nn]);
                    tg = add_k_edges(n,e,tg);                
    return tg, new_history;


def memoryless_instant_graph(vertex_dict,k,mode):
    from scipy.stats import norm
    if np.isscalar(k):
        kk = dict.fromkeys(vertex_dict.keys(),k);
    elif len(k)==2:
        dis = norm(loc=k[0],scale=k[1]);
        vv = map(lambda x: int(np.max([2, x])), dis.rvs(size=len(vertex_dict)));     
        kk = dict(zip(vertex_dict.keys(),vv))
    elif len(k)==len(vertex_dict):
        kk = dict(zip(vertex_dict.keys(),k))
        
    if mode=='simplicial':
        tgraph = nx.Graph()
        tgraph.add_nodes_from(vertex_dict.keys());
        new_history = []
        for n in vertex_dict:
            if np.random.rand()<=vertex_dict[n].act:
                nodes = vertex_dict.keys();
                nodes.remove(n)
                e = vertex_dict[n].new_collab(nodes,kk[n]);
                new_history.append(e);
                tgraph = add_clique(e,tgraph);
        return tgraph, new_history;
    
    if mode=='network':
        m = {};
        for n in vertex_dict:
            m[n] = kk[n]*(kk[n]-1)/2;
        tgraph = nx.Graph()
        tgraph.add_nodes_from(vertex_dict.keys());
        new_history = []
        for n in vertex_dict:
            if np.random.rand()<=vertex_dict[n].act:
                nodes = vertex_dict.keys();
                nodes.remove(n)
                neigh = random.sample(nodes,m[n]);
                for nn in neigh:
                    vertex_dict[n].new_edge(nn);
                    new_history.append([n,nn]);
                tgraph = add_k_edges(n,neigh,tgraph);
        return tgraph, new_history;

def temporal_graph_creation(N,T,k,act,mode,returnhist=False,verbose=False):
    tgraph = {};
    history = {}
    vertex_dict = {};
    for n in range(N):
        vertex_dict[n] = simplex_vertex(n,act[n])
    for t in range(T):
        if verbose == True and T%100==0:
            print T;
        tg, fh = memoryless_instant_graph(vertex_dict,k,mode);
        tgraph[t] = tg;
        history[t] = fh;
    if returnhist==True:
        return tgraph, history;
    else:
        return tgraph;

def memory_temporal_graph_creation(N,T,k,act,mode,alpha=1,returnhist=False,verbose=False):
    tgraph = {}
    history = {}
    vertex_dict = {}
    for n in range(N):
        vertex_dict[n] = simplex_vertex(n,act[n])
    for t in range(T):
        if verbose == True and T%100==0:
            print T;
        tg, fh = memory_instant_graph(vertex_dict,k,mode=mode,alpha=alpha);
        tgraph[t] = tg;
        history[t] = fh
    if returnhist==True:
        return tgraph, history;
    else:
        return tgraph;


def aggregate_graph(TG,T=None):
    if T==None:
        T = range(np.max(TG.keys()));
    w = {}
    for t in range(T):
        edges = TG[t].edges();
        for edge in edges:
            if edge not in w:
                w[edge] = 0;
            w[edge]+=1;
    G = nx.Graph();
    G.add_nodes_from(TG[0].nodes())
    G.add_edges_from(w.keys());
    nx.set_edge_attributes(G,'weight',w);
    return G;


def invert_k(M):
    return (1.0 + np.sqrt(1 + 8*M))/2.0;

def onion_decomposition(G):
    neigh = {}
    for n in G.nodes():
        neigh[n] = G.neighbors(n);
    nodes = G.nodes();
    D = G.degree();
    coreness = {}
    layerness = {}
    for n in nodes:
        coreness[n] = []
        layerness[n] = []
    core = 1 
    layer = 1
    while len(nodes)>0:
        thislayer = [n for n in nodes if D[n]<=core];
        for n in thislayer:
            coreness[n], layerness[n] = core, layer;
            for w in neigh[n]:
                if w in D:
                    D[w]-=1;
            nodes.remove(n);
            del D[n];
        layer+=1;
        if len(D)>0:
            if np.min(D.values())>=core+1:
                core = np.min(D.values());
    return coreness, layerness;
                
                
                
                
                
                
            

