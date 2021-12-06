#%%
import torch
from data.preprocess_dataset_qa import Model, Tokenizer, MODEL_NAME, DATASET_NAME


model = Model.from_pretrained(f'./{DATASET_NAME}_results/checkpoint-2500')
tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

# %%
# inputs = tokenizer('if you have a window size of five and you so you have to do these sort of 10 billion um, Softmax calculations before you work out what your gradient is', return_tensors='pt')
inputs = tokenizer('We normally sample as a small um like bunch order of approximately 32 or 64.', return_tensors='pt')
output = model(**inputs)
beg, end = (torch.argmax(output.start_logits), torch.argmax(output.end_logits) + 1)
answer_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0, beg:end],
                                                skip_special_tokens=True)
answer = tokenizer.convert_tokens_to_string(answer_tokens)
print(answer)
# %%
