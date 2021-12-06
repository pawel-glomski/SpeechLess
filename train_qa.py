from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from data.preprocess_dataset_qa import Model, MODEL_NAME, DATASET_NAME

dataset_train = load_from_disk(DATASET_NAME)
dataset_eval = load_from_disk(DATASET_NAME)

model = Model.from_pretrained(  #MODEL_NAME
    f'./{DATASET_NAME}_results/checkpoint-29000')

# for param in model.base_model.parameters():
#   param.requires_grad = False

training_args = TrainingArguments(output_dir=f'./{DATASET_NAME}_results',
                                  num_train_epochs=2,
                                  per_device_train_batch_size=12,
                                  per_device_eval_batch_size=16,
                                  warmup_steps=500,
                                  weight_decay=0.01,
                                  logging_dir='./logs',
                                  logging_steps=10,
                                  learning_rate=1e-4)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=dataset_train,
                  eval_dataset=dataset_eval)

trainer.train()
