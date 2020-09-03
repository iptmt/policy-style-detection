#!/bin/bash
for i in $(seq 0.0 0.05 1.0)
do
    echo $i >> ../log/yelp_mask.log
    cd ../src/baseline
    python main_mask.py yelp train $i >> /dev/null
    python main_mask.py yelp test $i >> /dev/null
    cd ../script/
    python style_shift.py yelp eval ../tmp/yelp.test.mask >> ../log/yelp_mask.log
done
