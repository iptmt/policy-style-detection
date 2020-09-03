import torch
from transformers import ElectraTokenizer, ElectraForPreTraining


sigmoid = torch.nn.Sigmoid()
device = torch.device("cuda:0")

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
model.to(device)
model.eval()

while True:
    text = input("input a sentence:")
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor(ids).long().to(device).unsqueeze(0)
    with torch.no_grad():
        outputs = sigmoid(model(input_ids=inputs)[0])
        scores = outputs.reshape(-1).cpu().detach().numpy().tolist()
        for t, s in zip(tokens, scores):
            print(t + "\t" + str(s))
