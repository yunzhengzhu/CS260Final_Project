import json
import sys

model = sys.argv[1]
f = open(f'../models/{model}/vocab.json', 'r')
vocab = json.load(f)
f.close()
vocab["<|bos|>"] = len(vocab)
vocab["<|pad|>"] = len(vocab)
print(len(vocab))
f = open(f'../models/{model}/vocab.json', 'w')
vocab = json.dump(vocab, f)
f.close()


