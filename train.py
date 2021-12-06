from datasets import load_from_disk
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
from data.preprocess_dataset import MAX_LENGTH, NUM_LABELS, ORIGINAL_TEXT_LABEL

# dataset_train = load_from_disk('cnn_dailymail_3.0.0_augumented_tokenized_train')
# dataset_eval = load_from_disk('glue_sst2_augumented_tokenized_validation')
dataset_train = load_from_disk('glue_sst2_augumented_tokenized_train')
dataset_eval = load_from_disk('glue_sst2_augumented_tokenized_validation')

model = DistilBertForTokenClassification.from_pretrained(
    'results6/checkpoint-2000',
    # 'distilbert-base-uncased',
    num_labels=NUM_LABELS)
# for param in model.base_model.parameters():
#   param.requires_grad = False

training_args = TrainingArguments(output_dir='./results7',
                                  num_train_epochs=2,
                                  per_device_train_batch_size=12,
                                  per_device_eval_batch_size=16,
                                  warmup_steps=500,
                                  weight_decay=0.01,
                                  logging_dir='./logs',
                                  logging_steps=10,
                                  learning_rate=1e-5)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=dataset_train,
                  eval_dataset=dataset_eval)

trainer.train()
# trainer.evaluate()
