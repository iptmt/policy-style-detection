#!/bin/bash

ds=$1

log_f="../log/$ds-attn.log"
if [ ! -f $log_f ]; then
    touch $log_f
fi

for i in $(seq 1.5 0.5 6)
do
    echo $i >> $log_f
    cd ../src/baseline
    python attn_score.py $ds inf $i >> /dev/null
    cd ../../script/
    python gap.py $ds ../tmp/$ds.test.mask.attn >> $log_f
done
