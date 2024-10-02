#!/bin/bash

# Set environment variables
export K=10
export THRESH=0.5
export THRESH_QUERY=0.5
export N_DOC_TO_SCORE=1000
export NPROBE=10
export OUT_SECOND_STAGE=100
export QUERIES_ID_FILE="aux_data/lotte/queries_id_lotte.tsv"
export INDEX_DIR_PATH="./index_dir/"
export ALLDOCLENS_PATH="aux_data/lotte/doclens_lotte.npy"


./build/perf_emvb
