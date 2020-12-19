#!/bin/bash

ds=$1

log_f="../log/$ds-freq.log"
if [ ! -f $log_f ]; then
    touch $log_f
fi

for j in $(seq 0 1 10)
do
    i=$((2**$j))
    echo $i >> $log_f
    cd ../src/baseline
    python freq_ratio.py $ds inf $i >> /dev/null
    cd ../../script/
    python gap.py $ds ../tmp/$ds.test.mask.ngrams >> $log_f
done