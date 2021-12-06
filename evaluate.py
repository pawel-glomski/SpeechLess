import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from speechless.readers import read_subtitles
from speechless.processing.tokenization import EditToken, sentence_segmentation
from data.preprocess_dataset import MAX_LENGTH, NUM_LABELS, ORIGINAL_TEXT_LABEL

model = DistilBertForTokenClassification.from_pretrained('results7/checkpoint-2500',
                                                         num_labels=NUM_LABELS)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


def summarize(transcript):
  sents = sentence_segmentation(transcript)
  sents_txt = [''.join([token.text for token in sent]) for sent in sents]
  text_tokenized = tokenizer(sents_txt,
                             is_split_into_words=True,
                             return_tensors='pt',
                             return_offsets_mapping=True).data
  last_split_end = 0
  text_tokenized_splits = []
  num_tokens = text_tokenized['input_ids'].shape[1]
  offsets = np.array(text_tokenized['offset_mapping'])[0]
  sent_starts = np.nonzero((offsets[:, 0] == 0) & (offsets[:, 1] != 0))[0]
  sent_starts = np.concatenate([sent_starts, [num_tokens]])  # add virtual last sentence
  for sent_idx, sent_start in enumerate(sent_starts):
    next_sent_start = sent_starts[sent_idx + 1] if sent_idx + 1 < len(sent_starts) else None
    if (sent_idx + 1) == len(sent_starts) or (next_sent_start - last_split_end) >= MAX_LENGTH:
      assert (sent_start - last_split_end) < MAX_LENGTH  # "<", not "<=", as sep_token will be added
      text_tokenized_splits.append({
          'attention_mask':
              torch.cat([
                  text_tokenized['attention_mask'][:, last_split_end:sent_start],
                  torch.tensor([[1]])
              ],
                        dim=1),
          'input_ids':
              torch.cat([
                  text_tokenized['input_ids'][:, last_split_end:sent_start],
                  torch.tensor([[tokenizer.sep_token_id]])
              ],
                        dim=1),
      })
      last_split_end = sent_start

  for split in text_tokenized_splits:
    split_result = model(**split)
    split_classified = torch.argmax(split_result.logits, axis=2)
    for split_idx, classified_tokens in enumerate(split_classified):
      split_ids = split['input_ids'][split_idx].numpy()
      split_text = tokenizer.convert_ids_to_tokens(split_ids, skip_special_tokens=True)
      summary_ids = split_ids[classified_tokens == ORIGINAL_TEXT_LABEL]
      summary = tokenizer.convert_ids_to_tokens(summary_ids, skip_special_tokens=True)
      removed_ids = split_ids[classified_tokens != ORIGINAL_TEXT_LABEL]
      removed = tokenizer.convert_ids_to_tokens(removed_ids, skip_special_tokens=True)
      print(
          # ' '.join(split_text), ' | ',
          ' '.join(summary),
          ' | ',
          removed)



summarize([
    EditToken(
        'Okay. Hello everyone. Um welcome back to this um well made course CS224N. I mean it. Okay so right at the end right at the end of last time I was I was just showing you a little',
        0, 1),
    EditToken('Um, so one of the famous, most famous steep learning people Yann Le Cun', 0, 1)
])
# summarize(read_subtitles('lecture/subs.en.vtt'))
