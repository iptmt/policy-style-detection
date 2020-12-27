python main_mask.py yelp inf
git checkout master
python main_rank.py yelp test
python main_insert.py yelp train
python main_insert.py yelp test
git checkout double-streams
cd ../script
