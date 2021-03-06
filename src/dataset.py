
import torch
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset, CoNLL2000Chunking
from torchtext.vocab import Vectors, GloVe, CharNGram

import numpy as np
import random
import logging
logger = logging.getLogger(__name__)


def conll2003_dataset(tag_type, batch_size, root='./conll2003', 
                          train_file='eng.train.txt', 
                          validation_file='eng.testa.txt',
                          test_file='eng.testb.txt',
                          convert_digits=True):
    """
    conll2003: Conll 2003 (Parser only. You must place the files)
    Extract Conll2003 dataset using torchtext. Applies GloVe 6B.200d and Char N-gram
    pretrained vectors. Also sets up per word character Field
    Parameters:
        tag_type: Type of tag to pick as task [pos, chunk, ner]
        batch_size: Batch size to return from iterator
        root: Dataset root directory
        train_file: Train filename
        validation_file: Validation filename
        test_file: Test filename
        convert_digits: If True will convert numbers to single 0's
    Returns:
        A dict containing:
            task: 'conll2003.' + tag_type
            iters: (train iter, validation iter, test iter)
            vocabs: (Inputs word vocabulary, Inputs character vocabulary, 
                    Tag vocabulary )
    """
    
    # Setup fields with batch dimension first
    inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, lower=True)

    inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", 
                                    batch_first=True)

    inputs_char = data.NestedField(inputs_char_nesting, 
                                    init_token="<bos>", eos_token="<eos>")
                        

    labels = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    fields = ([(('inputs_word', 'inputs_char'), (inputs_word, inputs_char))] + 
                [('labels', labels) if label == tag_type else (None, None) 
                for label in ['pos', 'chunk', 'ner']])

    # Load the data
    train, val, test = SequenceTaggingDataset.splits(
                                path=root, 
                                train=train_file, 
                                validation=validation_file, 
                                test=test_file,
                                separator=' ',
                                fields=tuple(fields))


    
    # Build vocab
    inputs_char.build_vocab(train.inputs_char, val.inputs_char, test.inputs_char)
    inputs_word.build_vocab(train.inputs_word, val.inputs_word, test.inputs_word, max_size=50000,
                        vectors=[GloVe(name='6B', dim='200'), CharNGram()])
    
    labels.build_vocab(train.labels)
  

    # Get iterators
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
                            (train, val, test), batch_size=batch_size, 
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    train_iter.repeat = False
    
    return {
        'task': 'conll2003.%s'%tag_type,
        'iters': (train_iter, val_iter, test_iter), 
        'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab) 
        }
    
