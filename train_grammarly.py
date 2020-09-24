import torch
import argparse
from models.recursive_models.recursive_eduembed import TreeRecursiveEduNN
from models.recursive_models.recursive_model import TreeRecursiveNN
from models.recursive_models.gcdc_model import GCDCModel
from models.recursive_models.ensemble_model import EnsembleModel
from models.recursive_models.recursive_tree_only import TreeRecursiveTreeOnlyNN

from datasets.tree_dataset import TreeDataset
from datasets.tree_dataset_ensemble import TreeDatasetEnsemble
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from glove.construct_glove import *
from train_helpers import *
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--embed_dim', type=int, default=50)
parser.add_argument('--glove_dim', type=int, default=300)
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=100)
args = parser.parse_args()


train_dataset_names = ['Clinton_train', 'Enron_train', 'Yahoo_train', 'Yelp_train']
test_dataset_names = ['Clinton_test', 'Enron_test', 'Yahoo_test', 'Yelp_test']
rec_model_names = ['rec_tree_only', 'rec_tree_nuc', 'rec_tree_rels', 'rec_tree_all']
ens_model_names = ['ens_tree_only', 'ens_tree_nuc', 'ens_tree_rels']
parseq_model_name = "parseq"

train_dataset_folder = "datasets/gcdc_trees/"
test_dataset_folder = "datasets/gcdc_trees/"
glove_path = "glove"

device = "cpu"
embed_dim = args.embed_dim
glove_dim = args.glove_dim
hidden_dim = args.hidden_dim
num_epochs = args.num_epochs
selected_model = args.model_name
run_id = args.run_id
lr=args.lr

loss=CrossEntropyLoss()

# Set up the dataset
print("Loading Glove embeddings")
glove, _, _ = load_glove(glove_path)

print("Constructing the datasets")
test_dataset_loaders = {}
if selected_model in rec_model_names:
    
    train_data_loader = DataLoader(TreeDataset(train_dataset_folder, train_dataset_names),
                               collate_fn=lambda x: x,
                               shuffle=True)
    for test_name in test_dataset_names:
        test_dataset_loaders[test_name] = DataLoader(TreeDataset(test_dataset_folder, [test_name]),
                                                   collate_fn=lambda x: x)
        
    if selected_model == 'rec_tree_only':
        model = TreeRecursiveTreeOnlyNN(hidden_dim)
    elif selected_model == 'rec_tree_nuc':
        model = TreeRecursiveNN(embed_dict_nuclearity, embed_dim, hidden_dim, use_relations=False)
    elif selected_model == 'rec_tree_rels':
        model = TreeRecursiveNN(embed_dict, embed_dim, hidden_dim, use_relations=True)
    else:
        model = TreeRecursiveEduNN(embed_dict, glove, embed_dim, glove_dim, hidden_dim, use_relations=True)
elif selected_model in ens_model_names:
    
    train_data_loader = DataLoader(TreeDatasetEnsemble(train_dataset_folder, train_dataset_names),
                                  collate_fn=lambda x: x,
                                  shuffle=True)
    for test_name in test_dataset_names:
        test_dataset_loaders[test_name] = DataLoader(TreeDatasetEnsemble(test_dataset_folder, [test_name]),
                                                   collate_fn=lambda x: x)
    if selected_model == 'ens_tree_only':
        recursive_model = TreeRecursiveTreeOnlyNN(hidden_dim)
    elif selected_model == 'ens_tree_nuc':
        recursive_model = TreeRecursiveNN(embed_dict_nuclearity, embed_dim, hidden_dim, use_relations=False)
    else:
        recursive_model = TreeRecursiveNN(embed_dict, embed_dim, hidden_dim, use_relations=True)
    sem_model = GCDCModel(embed_dict, glove, embed_dim, glove_dim, hidden_dim).to(device)
    model = EnsembleModel(recursive_model, sem_model, hidden_dim).to(device)
elif selected_model == parseq_model_name:
    train_data_loader = DataLoader(TreeDatasetEnsemble(train_dataset_folder, train_dataset_names),
                                  collate_fn=lambda x: x,
                                  shuffle=True)
    for test_name in test_dataset_names:
        test_dataset_loaders[test_name] = DataLoader(TreeDatasetEnsemble(test_dataset_folder, [test_name]),
                                                   collate_fn=lambda x: x)

    model = GCDCModel(embed_dict, glove, embed_dim, glove_dim, hidden_dim).to(device)
else:
    raise Exception("Unknown model name")


optimizer = Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model = model.train()
    print("***************************")
    print("EPOCH NUM %d" % epoch)
    cost_acc = 0
    for i, sample in enumerate(train_data_loader):
        cost_acc += train_step(sample, loss, optimizer, model)
        if (i % 100 == 0):
            print(cost_acc)
    print(cost_acc)
    
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': cost_acc
    },  "model_saves/" + selected_model + "_run_" + str(run_id))

model = model.eval()
    
for dataset_name, test_data_loader in test_dataset_loaders.items():
    acc, f1 = find_accuracy_F1(model, test_data_loader)
    print("Accuracy, F1 on ", dataset_name,  " are ", acc, f1)
    for label in range(3):
        recall = find_recall(model, test_data_loader, label)
        precision = find_precision(model, test_data_loader, label)
        if (precision != None):
            f1 = 2 * (precision * recall) / (precision + recall)
            print("F1 for label " + str(label) + " is " + str(f1))
        else:
            print("F1 for label " + str(label) + " is NA")
