import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_score, recall_score
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from matplotlib.ticker import MaxNLocator
from torch_geometric import seed_everything
from gnn import *

seed_everything(1234)
data = torch.load('/home/mellina/tfm/src/data/dataframe_all.pt')
data = T.ToUndirected()(data)
data["phenotype"].x = data["phenotype"].x.to(torch.float32)
data["gene"].x = data["gene"].x.to(torch.float32)

print(data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    edge_types=("phenotype", "related_to", "gene"),
    rev_edge_types=("gene", "rev_related_to", "phenotype"), 
)

train_data, val_data, test_data = transform(data)

model = HeteroGNN(hidden_channels=32, num_layers=6)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

edge_label_index = train_data["phenotype", "related_to", "gene"].edge_label_index
edge_label = train_data["phenotype", "related_to", "gene"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[-1, -1],
    edge_label_index=(("phenotype", "related_to", "gene"), edge_label_index),
    edge_label=edge_label,
    batch_size=4096,
    shuffle=True,
)

edge_label_index = val_data["phenotype", "related_to", "gene"].edge_label_index
edge_label = val_data["phenotype", "related_to", "gene"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[-1, -1],
    edge_label_index=(("phenotype", "related_to", "gene"), edge_label_index),
    edge_label=edge_label,
    batch_size=4096,
    shuffle=False,
)

losses = []
auprs = []

for epoch in range(1,11):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data["phenotype", "related_to", "gene"].edge_label_index)
        ground_truth = sampled_data["phenotype", "related_to", "gene"].edge_label
        loss = F.binary_cross_entropy(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    losses.append(total_loss/ total_examples)
    if epoch%1 == 0:
        preds = []
        ground_truths = []
        for sampled_data in tqdm.tqdm(val_loader):
            with torch.no_grad():
                sampled_data.to(device)
                ground_truth = sampled_data["phenotype", "related_to", "gene"].edge_label
                pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data["phenotype", "related_to", "gene"].edge_label_index)
                preds.append(pred)
                ground_truths.append(ground_truth)
        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)
        aupr = average_precision_score(ground_truth, pred)
        auprs.append(aupr)
        print(aupr)
        print(auc)
        
torch.save(model, '/home/mellina/tfm/src/data/model.pt')
losses = torch.Tensor(losses).cpu().numpy()
auprs = torch.Tensor(auprs).cpu().numpy()
epochs = range(1, len(losses)+1)

# Crear la gráfica
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
plt.grid()
# Plotear la función de pérdida en la primera columna
ax1.plot(epochs, losses, 'r', label='Función de pérdida')
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Error', fontsize=14)
ax1.tick_params(axis='both', labelsize=12)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

# Crear la segunda columna de la gráfica para el AUCPR
ax2 = ax1.twinx()
ax2.plot(epochs, auprs, 'b', label='AUCPR')
ax2.set_ylabel('AUCPR', fontsize=14)
ax2.tick_params(axis='both', labelsize=12)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax2.set_ylim(bottom=0, top=1)
ax1.legend(loc='upper right', fontsize=12)
ax2.legend(loc='upper left', fontsize=12)

# # Agregar título y leyenda de la gráfica
plt.title('Evolución de la función de pérdida y AUCPR', fontsize=16)
plt.tight_layout()
plt.savefig('/home/mellina/tfm/eval.png')

# cosas de test

edge_label_index = test_data["phenotype", "related_to", "gene"].edge_label_index
edge_label = test_data["phenotype", "related_to", "gene"].edge_label

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[-1, -1],
    edge_label_index=(("phenotype", "related_to", "gene"), edge_label_index),
    edge_label=edge_label,
    batch_size=2048,
    shuffle=False,
)

torch.save(test_loader, "/home/mellina/tfm/src/data/test.pt")