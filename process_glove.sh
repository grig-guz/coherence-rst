#!/bin/bash

mkdir glove
cd glove
wget http://nlp.stanford.edu/data/glove.42B.300d.zip
unzip glove.42B.300d.zip
cd ..

python construct_glove.py
