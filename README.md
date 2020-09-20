# Preparing Sequence Labeling Data for Transformers Is Hard

This repo has some utilities to align offset annotations (start,end) to tokenizer outputs, 
and to create pytorch datasets and dataloaders that handle padding and batching. 

The impetus for this repo is this [github issue](https://github.com/huggingface/transformers/issues/7019).
A blog post explaining our thinking around how to [best prepare sequence labeling data for use with pre-trained transformers](https://www.lighttag.io/blog/sequence-labeling-with-transformers/)
and another post, deriving [the implementation in this repo is here](https://lighttag.io/blog/sequence-labeling-with-transformers/example). 

This is a POC and maybe a work in progress. Issues, PRs and contributions welcome. 
The code is optimized for readability and clarity of thought. There is plenty of room for performance improvement, 
but not much of a case for it because compute time and memory are dominated by training. 

## Quick Example
If we have annotated data like this
```python
[{'annotations': [],
  'content': 'No formal drug interaction studies of Aranesp? have been '
             'performed.',
  'metadata': {'original_id': 'DrugDDI.d390.s0'}},
 {'annotations': [{'end': 13, 'label': 'drug', 'start': 6, 'tag': 'drug'},
                  {'end': 60, 'label': 'drug', 'start': 43, 'tag': 'drug'},
                  {'end': 112, 'label': 'drug', 'start': 105, 'tag': 'drug'},
                  {'end': 177, 'label': 'drug', 'start': 164, 'tag': 'drug'},
                  {'end': 194, 'label': 'drug', 'start': 181, 'tag': 'drug'},
                  {'end': 219, 'label': 'drug', 'start': 211, 'tag': 'drug'},
                  {'end': 238, 'label': 'drug', 'start': 227, 'tag': 'drug'}],
  'content': 'Since PLETAL is extensively metabolized by cytochrome P-450 '
             'isoenzymes, caution should be exercised when PLETAL is '
             'coadministered with inhibitors of C.P.A. such as ketoconazole '
             'and erythromycin or inhibitors of CYP2C19 such as omeprazole.',
  'metadata': {'original_id': 'DrugDDI.d452.s0'}},
 {'annotations': [{'end': 58, 'label': 'drug', 'start': 47, 'tag': 'drug'},
                  {'end': 75, 'label': 'drug', 'start': 62, 'tag': 'drug'},
                  {'end': 135, 'label': 'drug', 'start': 124, 'tag': 'drug'},
                  {'end': 164, 'label': 'drug', 'start': 152, 'tag': 'drug'}],
  'content': 'Pharmacokinetic studies have demonstrated that omeprazole and '
             'erythromycin significantly increased the systemic exposure of '
             'cilostazol and/or its major metabolites.',
  'metadata': {'original_id': 'DrugDDI.d452.s1'}}]
```
We can do this
```python
from sequence_aligner.labelset import LabelSet
from sequence_aligner.dataset import  TrainingDataset
from sequence_aligner.containers import TraingingBatch
import json
raw = json.load(open('./data/ddi_train.json'))
for example in raw:
    for annotation in example['annotations']:
        #We expect the key of label to be label but the data has tag
        annotation['label'] = annotation['tag']

from torch.utils.data import DataLoader
from transformers import BertForTokenClassification,AdamW
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(dataset.label_set.ids_to_label.values())
)
optimizer = AdamW(model.parameters(), lr=5e-6)

dataloader = DataLoader(
    dataset,
    collate_fn=TraingingBatch,
    batch_size=4,
    shuffle=True,
)
for num, batch in enumerate(dataloader):
    loss, logits = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_masks,
        labels=batch.labels,
    )
    loss.backward()
    optimizer.step()
