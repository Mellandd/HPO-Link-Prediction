from torch_geometric.explain import Explainer, CaptumExplainer
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
import torch_geometric.transforms as T
from torch_geometric.explain import ModelConfig
import captum
import pandas as pd
import torch_geometric.utils
import networkx as nx
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data = torch.load('/home/mellina/tfm/src/data/dataframe.pt')
data = T.ToUndirected()(data)
data["phenotype"].x = data["phenotype"].x.to(torch.float32)
data["gene"].x = data["gene"].x.to(torch.float32)
data = data.to(device)

class Classifier(torch.nn.Module):
    def forward(self, x_pheno: Tensor, x_gene: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_pheno[edge_label_index[0]]
        edge_feat_movie = x_gene[edge_label_index[1]]

        return torch.sigmoid((edge_feat_user * edge_feat_movie).sum(dim=-1))
        

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('phenotype', 'is_a', 'phenotype'): SAGEConv((-1, -1), hidden_channels),
                ('phenotype', 'related_to', 'gene'): SAGEConv((-1, -1), hidden_channels),
                ('gene', 'rev_related_to', 'phenotype'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)
        self.final = HeteroConv({
                ('phenotype', 'is_a', 'phenotype'): SAGEConv((-1, -1), hidden_channels),
                ('phenotype', 'related_to', 'gene'): SAGEConv((-1, -1), hidden_channels),
                ('gene', 'rev_related_to', 'phenotype'): SAGEConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
        self.classifier = Classifier()
        
    def encode(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = self.final(x_dict, edge_index_dict)
        return x_dict
    
    def decode(self, x_dict, edge_label_index):
        pred = self.classifier(
            x_dict["phenotype"],
            x_dict["gene"],
            #data["phenotype", "related_to", "gene"].edge_label_index,
            edge_label_index,
        )
        return pred

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = self.encode(x_dict, edge_index_dict)
        pred = self.decode(x_dict, edge_label_index)
        return pred
    
genes = '/home/mellina/tfm/src/data/exACGenes.csv'
phenotypes = '/home/mellina/tfm/src/data/phenotypes.csv'
gen_edges = '/home/mellina/tfm/src/data/phenotypes_to_genes.csv'
phen_edges = '/home/mellina/tfm/src/data/phenotype_edges.csv'

df_gen = pd.read_csv(genes, index_col='gene')
mapping_gene = {index: i for i, index in enumerate(df_gen.index.unique())}
mapping_gene_reverse = {i: index for i, index in enumerate(df_gen.index.unique())}
df_phen = pd.read_csv(phenotypes, index_col='Phenotypes')
mapping_phen = {index: i for i, index in enumerate(df_phen.index.unique())}
mapping_phen_reverse = {i: index for i, index in enumerate(df_phen.index.unique())}

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=False,
    edge_types=("phenotype", "related_to", "gene"),
    rev_edge_types=("gene", "rev_related_to", "phenotype"), 
)

train_data, val_data, test_data = transform(data)

model = torch.load('/home/mellina/tfm/src/data/model.pt')
model = model.to(device)

edge_label_index = train_data["phenotype", "related_to", "gene"].edge_label_index
index = torch.tensor([mapping_gene['SNCA'], mapping_phen['HP:0100315']])

explainer = Explainer(
    model=model,
    algorithm=CaptumExplainer('IntegratedGradients'),
    explanation_type='model',
    model_config=dict(
        mode='regression',
        task_level='edge',
        return_type='raw',
    ),
    node_mask_type='attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type='topk',
        value=20,
    ),
)
explanation = explainer(
    train_data.x_dict,
    train_data.edge_index_dict,
    index = index,
    edge_label_index = edge_label_index
)
print(f'Generated model explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
subgraph = explanation.get_explanation_subgraph()
print(subgraph)
#print(f"Feature importance plot has been saved to '{path}'")
#print(explanation.node_mask_dict)
#print(explanation.edge_mask_dict)
