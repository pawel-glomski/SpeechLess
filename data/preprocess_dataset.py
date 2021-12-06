import random
import spacy
import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

ORIGINAL_TEXT_LABEL = 0
IRRELEVANT_START_LABEL = 1
IRRELEVANT_PHRASE_LABEL = IRRELEVANT_START_LABEL
REPETITION_LABEL = IRRELEVANT_PHRASE_LABEL
RANDOM_START_LABEL = 2
NUM_LABELS = 2

IRRELEVANT_PHRASES = {
    'like', 'just', 'actually', 'basically', 'essentially', 'well', 'i mean', 'i guess',
    'i suppose', 'you know', 'you see', 'sort of', 'mhm', 'hmm', 'um', 'ah', 'uh', 'er', 'huh'
}
IRRELEVANT_PHRASES = (list(IRRELEVANT_PHRASES) +
                      [f'{a} {b}' for a in IRRELEVANT_PHRASES for b in IRRELEVANT_PHRASES])

IRRELEVANT_STARTS = {'right', 'great', 'nice', 'okay', 'ok', 'yea', 'yeah', 'so', 'now'}
IRRELEVANT_STARTS = (list(IRRELEVANT_STARTS) +
                     [f'{a} {b}' for a in IRRELEVANT_STARTS for b in IRRELEVANT_STARTS] +
                     [f'{a} {b}' for a in IRRELEVANT_STARTS for b in IRRELEVANT_PHRASES])

IRRELEVANT_PHRASES = [text.split() for text in IRRELEVANT_PHRASES]
IRRELEVANT_STARTS = [text.split() for text in IRRELEVANT_STARTS]

REPETITION_MAX_LEN = 3
RANDOM_START_MAX_LEN = 4

TEXT_COLUMN_NAME = 'text'
LABELS_COLUMN_NAME = 'labels'

MAX_LENGTH = 512


def augment_segment(sentence_segment, is_first_segment, other_docs):
  seg_text = [t.norm_ for t in sentence_segment]
  seg_labels = [ORIGINAL_TEXT_LABEL] * len(seg_text)

  # if is_first_segment and random.uniform(0, 1) <= 0.25:  # RANDOM_START
  #   # this can be combined with IRRELEVANT_START (applied after this), but not with REPETITION
  #   aug_length = random.randint(1, RANDOM_START_MAX_LEN)
  #   for other_doc in random.sample(other_docs, 32):
  #     long_sents = [s for s in other_doc.sents if len(s) > 2 * RANDOM_START_MAX_LEN]
  #     if len(long_sents) > 0:
  #       aug_source = random.choice(long_sents)
  #       aug_text = [token.norm_ for token in aug_source if not (token.is_punct or token.is_space)
  #                  ][:aug_length]
  #       seg_text[0:0] = aug_text
  #       seg_labels[0:0] = [RANDOM_START_LABEL] * len(aug_text)
  #       break
  # el
  if len(seg_text) >= REPETITION_MAX_LEN + 2 and random.uniform(0, 1) <= 0.5:  # REPETITION
    aug_start = random.randint(0,
                               len(seg_text) - 1 - REPETITION_MAX_LEN -
                               int(sentence_segment[-1].is_punct))  # account for "." at the end
    aug_length = random.randint(1, REPETITION_MAX_LEN)
    aug_text = seg_text[aug_start:aug_start + aug_length]
    aug_labels = [REPETITION_LABEL] * len(aug_text)
    seg_text[aug_start:aug_start] = aug_text
    seg_labels[aug_start:aug_start] = aug_labels

  if is_first_segment and random.uniform(0, 1) <= 0.4:  # IRRELEVANT_START
    aug_text = random.choice(IRRELEVANT_STARTS)
    aug_labels = [IRRELEVANT_START_LABEL] * len(aug_text)
    seg_text[0:0] = aug_text
    seg_labels[0:0] = aug_labels

  if not sentence_segment[-1].is_punct and random.uniform(0, 1) <= 0.8:  # IRRELEVANT_PHRASES
    aug_pos = len(seg_text)
    aug_phrase = random.choice(IRRELEVANT_PHRASES)
    aug_labels = [IRRELEVANT_PHRASE_LABEL] * len(aug_phrase)
    seg_text[aug_pos:aug_pos] = aug_phrase
    seg_labels[aug_pos:aug_pos] = aug_labels
  return (seg_text, seg_labels)


def prepare_preprocessing_fn(spacy_nlp, header_to_remove):

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
    aug_docs = []
    docs_labels = []
    for doc_idx, doc in enumerate(documents):
      other_docs = documents[:doc_idx] + documents[doc_idx + 1:]
      augumented_doc = []
      doc_labels = []
      for sentence in doc.sents:
        last_seg_end = 0
        for tok_idx, token in enumerate(sentence[:-1]):
          is_conjuction = token.tag_ in ['CC', 'IN']
          is_punct = token.norm_ == ','
          if is_conjuction or is_punct:
            if last_seg_end < tok_idx:
              segment = sentence[last_seg_end:tok_idx + 1]
              segment = [token for token in segment if not (token.is_punct or token.is_space)]
              if len(segment) > 0:
                seg_aug_text, seg_labels = augment_segment(segment, last_seg_end == 0, other_docs)
                augumented_doc.extend(seg_aug_text)
                doc_labels.extend(seg_labels)
              elif not is_punct:
                augumented_doc.append(token.norm_)
                doc_labels.append(ORIGINAL_TEXT_LABEL)
            last_seg_end = tok_idx + 1

        last_segment = [
            token for token in sentence[last_seg_end:len(sentence)]
            if not (token.is_punct or token.is_space)
        ]
        last_segment += [sentence[-1]] if sentence[-1].is_punct else []  # leave "." at the end
        last_seg_aug_text, last_seg_labels = augment_segment(last_segment, last_seg_end == 0,
                                                             other_docs)
        augumented_doc += last_seg_aug_text
        doc_labels += last_seg_labels
        assert len(augumented_doc) == len(doc_labels)
      aug_docs.append(augumented_doc)
      docs_labels.append(doc_labels)

    return {TEXT_COLUMN_NAME: aug_docs, LABELS_COLUMN_NAME: docs_labels}

  return preprocess_batch


def encode_labels(tokens, labels):
  token_labels = []
  for doc_labels, doc_offsets in zip(labels, tokens.offset_mapping):
    arr_offset = np.array(doc_offsets)
    doc_token_labels = np.ones(len(doc_offsets), dtype=int) * -100
    subtokens_starts = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
    doc_token_labels[subtokens_starts] = doc_labels[:np.count_nonzero(subtokens_starts)]
    token_labels.append(doc_token_labels.tolist())
  return token_labels


def prepare_tokenization_fn(model_tokenizer):

  def tokenize_batch(data_batch):
    documents, labels = data_batch[TEXT_COLUMN_NAME], data_batch[LABELS_COLUMN_NAME]
    tokenized_docs = model_tokenizer(documents,
                                     is_split_into_words=True,
                                     return_offsets_mapping=True,
                                     padding='max_length',
                                     truncation=True)
    tokenized_labels = encode_labels(tokenized_docs, labels)
    tokenized_docs.pop('offset_mapping')
    tokenized_docs[LABELS_COLUMN_NAME] = tokenized_labels
    return tokenized_docs

  return tokenize_batch


if __name__ == '__main__':
  nlp = spacy.load('en_core_web_md')
  nlp.add_pipe('sentencizer')
  tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

  with nlp.select_pipes(enable=['tok2vec', 'tagger', 'attribute_ruler', 'sentencizer']):
    datasets = [
        # ('cnn_dailymail', '3.0.0', 'article', ['train'], '(cnn)'),
        ('glue', 'sst2', 'sentence', ['train'], None)
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
