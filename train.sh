#!/bin/bash

python build_vocab.py -config  config/cluster77.yml
python train.py -config  config/cluster77.yml
