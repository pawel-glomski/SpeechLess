import random
import spacy
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AlbertTokenizerFast as Tokenizer
from transformers import AlbertForQuestionAnswering as Model  # pylint: disable=unused-import

MODEL_NAME = 'albert-base-v2'
DATASET_NAME = 'paraphrase_finding'

ORIGINAL_TEXT_LABEL = 0
IRRELEVANT_START_LABEL = 1
IRRELEVANT_PHRASE_LABEL = IRRELEVANT_START_LABEL
REPETITION_LABEL = IRRELEVANT_PHRASE_LABEL
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

SENT1_COLUMN_NAME = 'sent1'
SENT2_COLUMN_NAME = 'sent2'
LABEL_COLUMN_NAME = 'label'
TEXT_COLUMN_NAME = 'text'
START_POSITIONS_COLUMN_NAME = 'start_positions'
END_POSITIONS_COLUMN_NAME = 'end_positions'

MAX_LENGTH = 300

def augment_segment(sentence_segment, is_first_segment):
  seg_text = [t.norm_ for t in sentence_segment]
  seg_labels = [ORIGINAL_TEXT_LABEL] * len(seg_text)

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

def spacy_tokenize_sent(sent):
  return [token.norm_ for token in sent if not (token.is_punct or token.is_space)]

def prepare_preprocessing_fn(spacy_nlp):

  def preprocess_batch(data_batch):
    sents1 = list(spacy_nlp.pipe(data_batch[SENT1_COLUMN_NAME]))
    sents2 = list(spacy_nlp.pipe(data_batch[SENT2_COLUMN_NAME]))
    labels = [(label if isinstance(label, int) else label >= 3.25)
              for label in data_batch[LABEL_COLUMN_NAME]]

    texts = []
    starts = []
    ends = []
    for idx, (sent1, sent2, are_same) in enumerate(zip(sents1, sents2, labels)):
      sent1_text = spacy_tokenize_sent(sent1)
      sent2_text = spacy_tokenize_sent(sent2)
      sent2_text.append('.')

      if random.random() <= 0.8: # a random sentence will be at front
        other_sents = sents1[:idx] + sents1[idx + 1:] + sents2[:idx] + sents2[idx + 1:]
        prefix = spacy_tokenize_sent(random.choice(other_sents))
      else:
        prefix = []

      if are_same:  # the first sentence won't be finished
        sent1_len = random.randint(4, 5 + len(sent1_text) * 2 // 3)
        sent1_text = sent1_text[:sent1_len]
        are_same = are_same or sent1_len <= 3

      if random.random() <= 0.8:
        irrelevant_part = random.choice(IRRELEVANT_STARTS)
        if are_same:
          sent1_text += irrelevant_part
        else:
          prefix += sent1_text  # move sent1_text to the prefix
          sent1_text = irrelevant_part  # set sent1 to the irreleavant part
          are_same = True  # treat irreleavant part as a paraphrase of sent2

      texts.append(prefix + sent1_text + sent2_text)
      if are_same:
        starts.append(len(prefix))
        ends.append(len(prefix) + len(sent1_text) - 1)  # here ends are inclusive
      else:
        starts.append(0)
        ends.append(0)

    return {
        TEXT_COLUMN_NAME: texts,
        START_POSITIONS_COLUMN_NAME: starts,
        END_POSITIONS_COLUMN_NAME: ends
    }

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
    texts = data_batch[TEXT_COLUMN_NAME]
    starts = data_batch[START_POSITIONS_COLUMN_NAME]
    ends = data_batch[END_POSITIONS_COLUMN_NAME]
    tokenized_texts = model_tokenizer(texts,
                                      is_split_into_words=True,
                                      return_offsets_mapping=True,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=MAX_LENGTH)

    for idx, (start, end, offsets) in enumerate(zip(starts, ends, tokenized_texts.offset_mapping)):
      if start == end == 0:
        continue

      arr_offset = np.array(offsets)
      subtokens_starts = np.nonzero((arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0))[0]
      assert end + 1 < len(subtokens_starts)
      starts[idx] = subtokens_starts[start]
      ends[idx] = subtokens_starts[end + 1] - 1

    tokenized_texts.pop('offset_mapping')
    tokenized_texts[START_POSITIONS_COLUMN_NAME] = starts
    tokenized_texts[END_POSITIONS_COLUMN_NAME] = ends
    return tokenized_texts

  return tokenize_batch


if __name__ == '__main__':
  nlp = spacy.load('en_core_web_md')
  nlp.add_pipe('sentencizer')
  tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

  with nlp.select_pipes(enable=['tok2vec', 'tagger', 'attribute_ruler', 'sentencizer']):
    datasets = [('glue', 'mrpc', 'sentence1', 'sentence2', 'label', ['train']),
                ('glue', 'stsb', 'sentence1', 'sentence2', 'label', ['train']),
                ('glue', 'qqp', 'question1', 'question2', 'label', ['train'])]
    main_dataset = None
    for name, version, sent1_col_name, sent2_col_name, label_col_name, dataset_splits in datasets:
      for split in dataset_splits:
        dataset = load_dataset(name, version, split=split)
        columns_to_remove = set(dataset.features.keys())
        columns_to_remove -= {sent1_col_name, sent2_col_name, label_col_name}
        dataset = dataset.remove_columns(columns_to_remove)
        if sent1_col_name != SENT1_COLUMN_NAME:
          dataset = dataset.rename_column(sent1_col_name, SENT1_COLUMN_NAME)
        if sent1_col_name != SENT2_COLUMN_NAME:
          dataset = dataset.rename_column(sent2_col_name, SENT2_COLUMN_NAME)
        if label_col_name != LABEL_COLUMN_NAME:
          dataset = dataset.rename_column(label_col_name, LABEL_COLUMN_NAME)

        if len(dataset) > 9 * 1000:
          dataset = dataset.map(prepare_preprocessing_fn(nlp), batched=True, num_proc=9)
        else:
          dataset = dataset.map(prepare_preprocessing_fn(nlp), batched=True)

        dataset = dataset.remove_columns([LABEL_COLUMN_NAME])
        if main_dataset is None:
          main_dataset = dataset
        else:
          main_dataset = concatenate_datasets([main_dataset, dataset])

    main_dataset = main_dataset.map(prepare_tokenization_fn(tokenizer), batched=True, num_proc=9)
    main_dataset = main_dataset.remove_columns([TEXT_COLUMN_NAME])
    main_dataset = main_dataset.shuffle()
    main_dataset.save_to_disk(DATASET_NAME)
