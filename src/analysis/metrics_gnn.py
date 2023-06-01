import tqdm
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_score, recall_score
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from gnn import *

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_loader = torch.load('/home/mellina/tfm/src/data/test.pt')
model = torch.load('/home/mellina/tfm/src/data/model.pt')
model = model.to(device)
print(device)

preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(test_loader):
    with torch.no_grad():
        sampled_data.to(device)
        pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data["phenotype", "related_to", "gene"].edge_label_index)
        ground_truth = sampled_data["phenotype", "related_to", "gene"].edge_label
        preds.append(pred)
        ground_truths.append(ground_truth)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
pred_classes = np.ndarray.round(pred)

new_edges = []
true_edges = []
for i in range(len(ground_truth)):
    if ground_truth[i] == 0 and pred_classes[i] == 1:
        new_edges.append(pred[i])
    if ground_truth[i] == 1:
        true_edges.append(pred[i])

new_edges = np.array(new_edges)

# Métricas de TEST

print(f"F1: {f1_score(ground_truth, pred_classes)}")
print(f"Precision: {precision_score(ground_truth, pred_classes)}")
print(f"Recall: {recall_score(ground_truth, pred_classes)}")
print(f"ROC AUC: {roc_auc_score(ground_truth, pred)}")
print(f"PR AUC {average_precision_score(ground_truth, pred)}")
print(classification_report(ground_truth, pred_classes))
cm = confusion_matrix(ground_truth, pred_classes)
print(cm)
        
# Test

plt.figure()
plt.hist(new_edges, alpha=0.5, histtype='bar', ec='black')
plt.title("Distribución de la puntuación a las nuevas aristas")
plt.xlabel("Puntuación")
plt.ylabel("Número de aristas")
plt.tight_layout()
plt.savefig('/home/mellina/tfm/hist.png')

plt.figure()
plt.hist(true_edges, alpha=0.5, histtype='bar', ec='black')
plt.title("Distribución de la puntuación de las aristas del grafo")
plt.xlabel("Puntuación")
plt.ylabel("Número de aristas")
plt.savefig('/home/mellina/tfm/hist_true.png')
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['No', 'Yes'])
disp.plot()
plt.savefig('/home/mellina/tfm/cm.png')

