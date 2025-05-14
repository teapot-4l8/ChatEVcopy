# Foundation packages
import torch
import pandas as pd
import numpy as np

# packages for data processing
import utils
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")


# input data
occ, dur, vol, prc, adj, col, dis, weather, inf = utils.read_data()
zone = 42
timestamp = 1024  # note: timestamp > 12
input_prompt = utils.prompting(zone, timestamp, inf, occ, prc, weather)
target = utils.output_template(np.round(occ.iloc[timestamp+6, zone], decimals=4))
print(input_prompt)

model, tokenizer, config = utils.load_llm()  # load model and tokenizer

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=device)
messages = [
    {"role": "user", "content": input_prompt}
]

outputs = pipe(
    messages,
    max_new_tokens=128,
    do_sample=True
)
print(outputs[0]["generated_text"][-1])
print('Groundtruth =', target)
