python generate.py -d=yelp -m=train -t=rnn -trf=../tmp/yelp.train.mask -def=../tmp/yelp.dev.mask
python generate.py -d=yelp -m=train -t=rnn-attn -trf=../tmp/yelp.train.mask -def=../tmp/yelp.dev.mask
python generate.py -d=yelp -m=train -t=mlm -trf=../tmp/yelp.train.mask -def=../tmp/yelp.dev.mask

python generate.py -d=yelp -m=inf -t=rnn -tef=../tmp/yelp.test.mask
python generate.py -d=yelp -m=inf -t=rnn-attn -tef=../tmp/yelp.test.mask
python generate.py -d=yelp -m=inf -t=mlm -tef=../tmp/yelp.test.mask
