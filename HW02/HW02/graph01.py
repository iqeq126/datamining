import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def ex01():
    # reading the GML file
    G = nx.read_gml('karate.gml', label='id')
    print(G.nodes())
    nx.draw(G)
    plt.show()
    print(nx.info(G))
    ## Adjacency matrix
    a_matrix = nx.adjacency_matrix(G)
    ## Imporving the graph by trying out different layouts
    graph_layouts = [nx.circular_layout(G), nx.random_layout(G), nx.spring_layout(G), nx.spectral_layout(G)]

    for graph in graph_layouts:
        plt.subplots(figsize=(15, 15))
        nx.draw(G, with_labels=True, node_size=700, node_color="#e1575c", edge_color='#363847', pos=graph)
        plt.show()

    ## Best visualisation
    nx.draw(G, with_labels=True, node_size=700, node_color="#e1575c", edge_color='#363847', pos=nx.spring_layout(G))
    plt.title("Zacary's Karate Club: Spring Layout", fontsize=20)
    plt.axis('off')
    plt.show()

    ## Computing weighted eigenvector centrality
    centrality_nx = nx.eigenvector_centrality(G)
    l_eigen_cent = list(centrality_nx.values())

    ## Finding the student node in graph who has the largest cenrtality value (most emails sent)
    (largest_hub, degree) = sorted(centrality_nx, key=centrality_nx.get, reverse=True)[:2]
    print(largest_hub)

    df_eig_cent = pd.DataFrame(centrality_nx.items(), columns=['person', 'degree_centrality'])
    print(df_eig_cent.sort_values('degree_centrality', ascending=False).head())

    ## Degree Centrality
    ## The function degree_centrality(G) is available in networkx.algorithms.centrality
    deg_cent = nx.algorithms.centrality.degree_centrality(G)
    l_deg_cent = list(deg_cent.values())
    print(l_deg_cent)

    ## The output is dictionary of nodes with degree centrality as the value
    print("Degree centrality:")
    # print(deg_cent)

    df_deg_cent = pd.DataFrame(deg_cent.items(), columns=['person', 'degree_centrality'])
    print(df_deg_cent.sort_values('degree_centrality', ascending=False).head())

def ex02():
    G = nx.read_gml('adjnoun.gml')
    print(G.nodes)

    nx.draw(G)
    plt.show()
    print(nx.info(G))

    ## Adjacency matrix
    a_matrix = nx.adjacency_matrix(G)
    graph_layouts = [nx.circular_layout(G), nx.random_layout(G), nx.spring_layout(G), nx.spectral_layout(G)]
    for graph in graph_layouts:
        plt.subplots(figsize=(15, 15))
        nx.draw(G, with_labels=True, node_size=700, node_color="#e1575c", edge_color='#363847', pos=graph)
        plt.show()

    ## Best visualisation
    nx.draw(G, with_labels=True, node_size=700, node_color="#e1575c", edge_color='#363847', pos=nx.spring_layout(G))
    plt.title("Nouns", fontsize=20)
    plt.axis('off')
    plt.show()