#!/bin/bash

python generate_test_file.py
python translate.py -model model/cluster77_noaug_step_100000.pt -src re_index_uspto_test_out_top10.txt -output output/output/pred_test_top3_norank.targets.txt     -batch_size 1024 -replace_unk -max_length 200 -beam_size 3  -n_best 3  -gpu 0
python rank_the_output.py
python cal_top_k_maxfrag_file.py
