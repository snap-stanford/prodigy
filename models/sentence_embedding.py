'''
    Simple class to embed sentences using a pre-trained lanugage model.
'''

from sentence_transformers import SentenceTransformer
from torch import from_numpy, zeros, stack
import os
from time import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#cache_folder_bert = os.environ.get("CACHE_FOLDER_BERT", "./FSdatasets/arxiv/sbert")


class SentenceEmb:
    def __init__(self, model, device, dummy=False, use_cache=True, cache_folder=None):
        '''

        :param model: The bert model to use.
        :param device:
        :param dummy:
        :param use_cache: Set to True if you want the model to cache sentences in memory for faster inference.
        '''
        #  https://www.sbert.net/docs/pretrained_models.html
        assert cache_folder is not None, "cache_folder is a required argument"
        if dummy:
            self.model = None
        else:
            self.model = SentenceTransformer(model, cache_folder=cache_folder, device=device)
        if use_cache:
            self.cache = {}
        else:
            self.cache = None
        self.device = device

    def get_sentence_embeddings(self, sentence_list):
        '''
        :param emb_list: either list of sentences or sentence (str)
        :return:
        '''

        if not isinstance(sentence_list, list):
            sentence_list = [sentence_list]
        if self.model is None:
            return zeros(len(sentence_list), 768).float()
        t1 = time()
        unknown_sentences = [sent for sent in sentence_list if sent not in self.cache]
        if len(unknown_sentences) > 0:
            unknown_embeddings = self.model.encode(unknown_sentences, convert_to_tensor=True)
            for i, sent in enumerate(unknown_sentences):
                self.cache[sent] = unknown_embeddings[i].cpu()
        t2 = time()
        return stack([self.cache[sent] for sent in sentence_list])

