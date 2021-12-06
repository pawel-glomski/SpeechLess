import spacy
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

SPOKEN_TEXT_LABEL = 0
WRITTEN_TEXT_LABEL = 1

TEXT_COLUMN_NAME = 'text'
LABELS_COLUMN_NAME = 'labels'

MAX_LENGTH = 512


def prepare_preprocessing_fn(spacy_nlp, header_to_remove, label):

  def preprocess_batch(data_batch):
    documents = []
    for doc in data_batch[TEXT_COLUMN_NAME]:
      doc = doc.lower()
      if header_to_remove is not None:
        start_offset = doc.find(header_to_remove)
        if start_offset >= 0:
          doc = doc[start_offset + len(header_to_remove):]
      documents.append(doc)
    documents = list(spacy_nlp.pipe(documents))

    sents = []
    for doc in documents:
      for sentence in doc.sents:
        sents += [token for token in sentence if not (token.is_punct or token.is_space)]
    labels = [label] * len(sents)

    return {TEXT_COLUMN_NAME: sents, LABELS_COLUMN_NAME: labels}

  return preprocess_batch


def prepare_tokenization_fn(model_tokenizer):

  def tokenize_batch(data_batch):
    documents, labels = data_batch[TEXT_COLUMN_NAME], data_batch[LABELS_COLUMN_NAME]
    tokenized_docs = model_tokenizer(documents,
                                     is_split_into_words=True,
                                     padding='max_length',
                                     truncation=True)
    tokenized_docs[LABELS_COLUMN_NAME] = labels
    return tokenized_docs

  return tokenize_batch


if __name__ == '__main__':
  nlp = spacy.load('en_core_web_md')
  nlp.add_pipe('sentencizer')
  tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

  with nlp.select_pipes(enable=['tok2vec', 'tagger', 'attribute_ruler', 'sentencizer']):
    datasets = [
        ('cnn_dailymail', '3.0.0', 'article', ['train'], '(cnn)', WRITTEN_TEXT_LABEL),
        # ('glue', 'sst2', 'sentence', ['train'], None, WRITTEN_TEXT_LABEL)
    ]
    for name, version, text_column_name, dataset_splits, header in datasets:
      for split in dataset_splits:
        dataset = load_dataset(name, version, split=split)
        columns_to_remove = list(dataset.features.keys())
        columns_to_remove.remove(text_column_name)
        dataset = dataset.remove_columns(columns_to_remove)
        dataset = dataset.rename_column(text_column_name, TEXT_COLUMN_NAME)
        dataset = dataset.map(prepare_preprocessing_fn(nlp, header), batched=True, num_proc=9)
        dataset.save_to_disk(f'{name}_{version}_augumented_{split}')

        dataset = dataset.map(prepare_tokenization_fn(tokenizer), batched=True, num_proc=9)
        dataset = dataset.remove_columns([TEXT_COLUMN_NAME])
        dataset.save_to_disk(f'{name}_{version}_augumented_tokenized_{split}')
