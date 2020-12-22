#!/bin/bash

ds=$1

log_f="../log/$ds-tfidf.log"
if [ ! -f $log_f  ]; then
    touch $log_f
fi

for i in $(seq 0.25 0.0611111 0.8)
do
    echo $i >> $log_f
    cd /home/zaracs/workspace/baselines/tagger-generator/tag-and-generate-data-prep
    python src/run.py --data_pth ../data/$ds.tsv --outpath ../data/ --style_0_label P_9 --style_1_label P_0 --is_unimodal True --thresh $i

    cd /home/zaracs/workspace/policy-style-detection/script
    python eval_tg.py $ds

    python gap.py $ds ../tmp/$ds.test.mask.tfidf >> $log_f
done
