#!/bin/bash

ds=$1

log_f="../log/$ds-ours.log"
if [ ! -f $log_f ]; then
    touch $log_f
fi

for i in $(seq 0.8 0.1 0.8)
do
    echo $i >> $log_f
    cd ../src
    python main_mask.py $ds inf $i >> /dev/null
    cd ../script/
    python gap.py $ds ../tmp/$ds.test.mask >> $log_f
done
