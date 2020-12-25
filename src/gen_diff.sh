#!/bin/bash
python generate.py -d=yelp -m=train -t=rnn -trf=../tmp/yelp.train.mask -def=../tmp/yelp.dev.mask
python generate.py -d=yelp -m=train -t=rnn-attn -trf=../tmp/yelp.train.mask -def=../tmp/yelp.dev.mask
python generate.py -d=yelp -m=train -t=mlm -trf=../tmp/yelp.train.mask -def=../tmp/yelp.dev.mask

python generate.py -d=yelp -m=inf -t=rnn -tef=../out/ours/yelp_test_rnn.tsf
python generate.py -d=yelp -m=inf -t=rnn-attn -tef=../out/ours/yelp_test_rnn-attn.tsf
python generate.py -d=yelp -m=inf -t=mlm -tef=../out/ours/yelp_test_mlm.tsf

python generate.py -d=gyafc -m=train -t=rnn -trf=../tmp/gyafc.train.mask -def=../tmp/gyafc.dev.mask
python generate.py -d=gyafc -m=train -t=rnn-attn -trf=../tmp/gyafc.train.mask -def=../tmp/gyafc.dev.mask
python generate.py -d=gyafc -m=train -t=mlm -trf=../tmp/gyafc.train.mask -def=../tmp/gyafc.dev.mask

python generate.py -d=gyafc -m=inf -t=rnn -tef=../out/ours/gyafc_test_rnn.tsf
python generate.py -d=gyafc -m=inf -t=rnn-attn -tef=../out/ours/gyafc_test_rnn-attn.tsf
python generate.py -d=gyafc -m=inf -t=mlm -tef=../out/ours/gyafc_test_mlm.tsf