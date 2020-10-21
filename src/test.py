import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from bleurt import score

checkpoint = "/home/zaracs/ckpts/bleurt-base-128"
references = ["we were happy to go there."]
candidates = ["we were not happy to go there."]
# candidates = ["It is our pleasure to go there."]

scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references, candidates)
assert type(scores) == list and len(scores) == 1
print(scores)
