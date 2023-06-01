import networkx as nx
import obonet
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch


# Leemos la ontología directamente de la página web
url = 'https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo'
graph = obonet.read_obo(url)

descr = []
for node in graph.nodes():
    if 'def' in graph.nodes[node]:
        descr.append(graph.nodes[node]['def'])
    else:
        descr.append(' ')

for node in graph.nodes():
    node_attrs = dict(graph.nodes[node])  # create a copy of the attributes
    for attr in node_attrs.keys():
        del graph.nodes[node][attr]
        
#nx.write_gexf(graph, '/home/mellina/tfm/src/data/original_graph.gexf')

phen = [x[0] for x in graph.edges]
is_a = [x[1] for x in graph.edges]

df = pd.DataFrame(graph.nodes(), columns=['Phenotypes'])
df['Definition'] = descr
df.to_csv('/home/mellina/tfm/src/data/phenotypes.csv', index=False)

df2 = pd.DataFrame({'Phenotype': phen, 'is_a': is_a})
df2.to_csv('/home/mellina/tfm/src/data/phenotype_edges.csv', index=False)