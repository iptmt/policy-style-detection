import os
from nltk.tokenize import TweetTokenizer, word_tokenize
import nltk
from fio import read, write, read_csv



all_0 = read_csv("../out/tdrg/yelp_all_model_prediction_ref0.csv")
all_1 = read_csv("../out/tdrg/yelp_all_model_prediction_ref1.csv")

lines_del_0 = [row["Source"] + "\t" + row["BERT_DEL"] + "\t" + "1" for row in all_0]
lines_del_1 = [row["Source"] + "\t" + row["BERT_DEL"] + "\t" + "0" for row in all_1]
lines_B_GST = lines_del_0 + lines_del_1

write("../out/tdrg/yelp_test_B_GST.tsf", lines_B_GST)

lines_ret_0 = [row["Source"] + "\t" + row["BERT_RET_TFIDF"] + "\t" + "1" for row in all_0]
lines_ret_1 = [row["Source"] + "\t" + row["BERT_RET_TFIDF"] + "\t" + "0" for row in all_1]
lines_G_GST = lines_ret_0 + lines_ret_1

write("../out/tdrg/yelp_test_G_GST.tsf", lines_G_GST)