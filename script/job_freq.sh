#!/bin/bash
for i in $(seq 0 5 100)
do
    echo $i >> ../log/yelp_freq.log
    cd ../src/baseline
    python freq_ratio.py yelp inf $i >> /dev/null
    cd ../../script/
    python style_shift.py yelp eval ../tmp/yelp.test.mask.ngrams >> ../log/yelp_freq.log
done
