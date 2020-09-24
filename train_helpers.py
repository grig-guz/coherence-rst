import torch
from sklearn.metrics import accuracy_score, f1_score

embed_dict = {'Topic-Change_Nucleus' : 0, 'Topic-Change_Satellite' : 1, 'Topic-Comment_Nucleus' : 2,
                'Topic-Comment_Satellite' : 3, 'Manner-Means_Nucleus' : 4,  'Manner-Means_Satellite' : 5,
                'Comparison_Nucleus': 6, 'Comparison_Satellite' : 7, 'Evaluation_Nucleus': 8,
                'Evaluation_Satellite': 9, 'Summary_Nucleus': 10, 'Summary_Satellite': 11,
                 'Condition_Nucleus': 12, 'Condition_Satellite': 13, 'Enablement_Nucleus': 14,
                 'Enablement_Satellite': 15, 'Cause_Nucleus': 16, 'Cause_Satellite': 17,
                 'Temporal_Nucleus': 18, 'Temporal_Satellite': 19, 'Explanation_Nucleus': 20,
                 'Explanation_Satellite': 21, 'Background_Nucleus': 22, 'Background_Satellite': 23,
                 'Contrast': 24, 'Joint': 25, 'Same-Unit':26,
                 'Attribution_Nucleus':27, 'Attribution_Satellite':28, 'Elaboration_Nucleus':29,
                 'Elaboration_Satellite':30, 'TextualOrganization': 31}


embed_dict_nuclearity = {'Nucleus': 0, 'Satellite':1}

for key in embed_dict:
    embed_dict[key] = torch.LongTensor([embed_dict[key]])

for key in embed_dict_nuclearity:
    embed_dict_nuclearity[key] = torch.LongTensor([embed_dict_nuclearity[key]])


model_load_paths = {
                    "TreeRecursiveNN": "model_saves/discourse_plain/epoch_11",
                    "TreeRecursiveEduNN": "model_saves/discourse_edus/epoch_10",
                    "RecursiveGCDCEnsemble": "model_saves/models_to_evaluate/semantic_4",
                    "TreeRecursiveSemanticOnlyNN": "model_saves/semantic_only/epoch_4",
                    "ParSeq": "model_saves/models_to_evaluate/parseq"
                    }



def find_accuracy_F1(model, test_data_loader):
    incorrect_count = 0
    y_pred, y_true = [], []
    for i, sample in enumerate(test_data_loader):
        X, y = sample[0]
        out, _ = model(X)
        y_pred.append(torch.argmax(out[0]))
        y_true.append(y[0])
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return acc, f1

def train_step(sample, loss, optim, model):
    X, y = sample[0]
    optim.zero_grad()
    out, _ = model(X)
    cost = loss(out, y)
    cost.backward()
    optim.step()
    return cost.item()

def find_recall(model, test_data_loader, label):
    label_count = 0
    correct_label_count = 0
    for i, sample in enumerate(test_data_loader):
        X, y = sample[0]
        out, _ = model(X)
        for i in range(out.size()[0]):
            if (y[i] == label and y[i] == torch.argmax(out[i])):
                correct_label_count += 1
                label_count += 1
            elif (y[i] == label):
                label_count += 1
    print("Recall for class ",  label,  " is ", correct_label_count / label_count)
    return correct_label_count / label_count

def find_precision(model, test_data_loader, label):
    label_count = 0
    correct_label_count = 0
    for i, sample in enumerate(test_data_loader):
        X, y = sample[0]
        out, _ = model(X)
        for i in range(out.size()[0]):
            if (y[i] == label and y[i] == torch.argmax(out[i])):
                correct_label_count += 1
                label_count += 1
            elif (torch.argmax(out[i]) == label):
                label_count += 1
    if label_count == 0:
        print("Precision NA")
    else:
        print("Precision for class ",  label,  " is ", correct_label_count / label_count)
