# Neural RST-based Evaluation of Discourse Coherence

This repository contains the code for "Neural RST-based Evaluation of Discourse Coherence" by Grigorii Guz, Peyman Bateni, Darius Muglich, and Prof. Guiseppe Carenini. The paper has been accepted to and will be presented at AACL-IJCNLP 2021. For a preprint ArXiv copy of the paper, please visit https://arxiv.org/pdf/2009.14463.pdf.

## Package Requirements
You can easily replicate our enviroment along with the required dependencies using ```conda create --name <env> --file requirements.txt```. All dependencies and corresponding versions have been provided in ```requirements.txt```

## Compute Requirements
The results reported in the paper were trained using 5 CPU-nodes and 30Gb of RAM. However, much lower compute power should also be sufficient when training as we also experiments on local machines with less that 16Gb of RAM. Note that there are no GPU requirements.

## Setup & Training/Testing
0. Setup and activate the working environment according to the instruction under Package Requirements.
1. Clone or download this repository.
2. Set up glove embeddings by running ```bash process_glove.sh```. This will download the appropriate embeddings from http://nlp.stanford.edu/data/glove.42B.300d.zip and then, it will itself run -u train_grammarly.py --model_name ens_tree_only --run_id 0 --lr 0.0001 --num_epochs 2 --embed_dim 50 --glove_dim 300 --hidden_dim 100. Depending on your download and compute speed, this step should take between 30 minutes to 4 hours.
2. Request permission and download the GCDC dataset from https://github.com/aylai/GCDC-corpus. Upon email confirmation of access to the GCDC dataset, please reach out to any of the first-authors of this paper via email to recieve the RST-parsed dataset, which we will happily provide for you. Note that this RST-parsing has been very computationally intensive, having used multiple AWS EC2 instances over the span of a week. If you are familiar with the CODRA parser, and would like to perform this parsing yourself, we've provided both example fixing and parsing scripts under the ```dataset``` folder. We can provide additional intructions for doing so if viewers are interested (please open an issue :D). Otherwise, we will provide the already parsed examples.
3. One set-up has been complete, you can train/test each model variation by: 
    ```python -u train_grammarly.py --model_name <model_name> --run_id <run_id> --lr 0.0001 --num_epochs 2 --embed_dim 50 --glove_dim 300 --hidden_dim 100```
    Note that you can change any of the hyperparameters to try other variations too.
    Alternatively, if you would to run all experiments at once, scripts for each model variation have been provided under ```run_scripts```.

**Model accuracies (averaged across 1000 independent runs)**

| Model       | Clinton Sub-Dataset | Enron Sub-Dataset | Yahoo Sub-Dataset | Yelp Sub-Dataset | Overall |
| ---         | ---                 | ---               | ---               | ---              | --- |
| rec_tree_only | 55.33±0.00 | 44.39±0.00 | 38.02±0.00 | 54.82±0.00 | 48.14±0.00 |
| rec_tree_nuc | 53.74±0.14 | 44.67±0.07 | 44.61±0.09 | 53.76±0.11 | 49.20±0.07 |
| rec_tree_rels | 54.07±0.10 | 43.99±0.07 | 49.39±0.10 | 54.39±0.12 | 50.46±0.05 |
| rec_tree_all | 55.70±0.08 | 53.86±0.11 | 50.92±0.13 | 51.70±0.16 | 53.04±0.09 |
| ---         | ---                 | ---               | ---               | ---              | --- |
| parseq | 61.05±0.13 | 54.23±0.10 | 53.29±0.14 | 51.76±0.21 | 55.09±0.09 |
| ---         | ---                 | ---               | ---               | ---              | --- |
| ens_tree_only | 61.12±0.13 | 54.20±0.12 | 52.87±0.16 | 51.52±0.22 | 54.93±0.10 |
| ens_tree_nuc | 60.82±0.13 | 54.01±0.10 | 52.92±0.15 | 51.63±0.24 | 54.85±0.10 |
| ens_tree_rels | 61.17±0.12 | 53.99±0.10 | 53.99±0.14 | 52.40±0.21 | 55.39±0.09 |

**Model F1-scores (averaged across 1000 independent runs)**

| Model       | Clinton Sub-Dataset | Enron Sub-Dataset | Yahoo Sub-Dataset | Yelp Sub-Dataset | Overall |
| ---         | ---                 | ---               | ---               | ---              | --- |
| rec_tree_only | 39.42±0.00 | 27.29±0.00 | 20.95±0.00 | 38.82±0.00 | 31.62±0.00 |
| rec_tree_nuc | 39.20±0.03 | 30.81±0.16 | 35.67±0.18 | 39.93±0.08 | 36.40±0.09 |
| rec_tree_rels | 41.08±0.07 | 31.21±0.13 | 41.97±0.14 | 42.27±0.09 | 39.13±0.08 |
| rec_tree_all | 45.90±0.12 | 44.33±0.16 | 43.85±0.18 | 43.13±0.10 | 44.30±0.08 |
| ---         | ---                 | ---               | ---               | ---              | --- |
| parseq | 52.12±0.21 | 44.90±0.15 | 46.22±0.18 | 43.36±0.09 | 46.65±0.10 |
| ---         | ---                 | ---               | ---               | ---              | --- |
| ens_tree_only | 52.35±0.22 | 44.92±0.16 | 45.48±0.22 | 43.70±0.11 | 46.61±0.11 |
| ens_tree_nuc | 51.90±0.22 | 44.76±0.14 | 45.48±0.22 | 43.83±0.13 | 46.49±0.10 |
| ens_tree_rels | 52.42±0.19 | 44.69±0.15 | 46.88±0.17 | 43.94±0.09 | 46.98±0.09 |

## Citing this repository/paper
```
@inproceedings{guz-etal-2020-neural,
    title = "Neural {RST}-based Evaluation of Discourse Coherence",
    author = "Guz, Grigorii  and
      Bateni, Peyman  and
      Muglich, Darius  and
      Carenini, Giuseppe",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.aacl-main.67",
    pages = "664--671",
    abstract = "This paper evaluates the utility of Rhetorical Structure Theory (RST) trees and relations in discourse coherence evaluation. We show that incorporating silver-standard RST features can increase accuracy when classifying coherence. We demonstrate this through our tree-recursive neural model, namely RST-Recursive, which takes advantage of the text{'}s RST features produced by a state of the art RST parser. We evaluate our approach on the Grammarly Corpus for Discourse Coherence (GCDC) and show that when ensembled with the current state of the art, we can achieve the new state of the art accuracy on this benchmark. Furthermore, when deployed alone, RST-Recursive achieves competitive accuracy while having 62{\%} fewer parameters.",
}
```
