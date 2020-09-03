#!/bin/bash
python main_mask.py yelp train
python main_mask.py yelp test
python main_rank.py yelp test
python main_insert.py yelp train
