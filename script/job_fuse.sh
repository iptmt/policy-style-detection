#!/bin/bash
for i in $(seq 0 0.5 10.0)
do
    echo $i >> ../log/yelp_fuse.log
    cd ../src/baseline
    python fusion.py yelp $i >> /dev/null
    cd ../../script/
    python style_shift.py yelp eval ../tmp/yelp.test.mask.fuse >> ../log/yelp_fuse.log
done
