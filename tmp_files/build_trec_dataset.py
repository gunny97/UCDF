#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""
import glob
import json
import logging
import pickle
import time
from xmlrpc.client import boolean
import zlib
from typing import List, Tuple, Dict, Iterator
import random
import csv
import hydra
import numpy as np
import pandas as pd
import os
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F
from dpr.utils.data_utils import RepTokenSelector
from dpr.data.qa_validation import calculate_matches, calculate_chunked_matches, calculate_matches_from_meta
from dpr.data.retriever_data import KiltCsvCtxSrc, TableChunk
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from dpr.models import init_biencoder_components
from dpr.models.biencoder import (
    BiEncoder,
    _select_span_with_token,
)
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser(description='build positive/negative dataset paper')

logger = logging.getLogger()
setup_logger(logger)

def get_all_passages(ctx_sources):
    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)
        logger.info("Loaded ctx data: %d", len(all_passages))
        print(len(all_passages))

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages

def iterate_encoded_files(vector_files: list, path_id_prefixes: List = None) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc

def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: str = None,
    selector: RepTokenSelector = None,
) -> T:
    n = len(questions)
    query_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    batch_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token) for q in batch_questions
                    ]
                else:
                    batch_tensors = [tensorizer.text_to_tensor(" ".join([query_token, q])) for q in batch_questions]
            elif isinstance(batch_questions[0], T):
                batch_tensors = [q for q in batch_questions]
            else:
                batch_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]

            # TODO: this only works for Wav2vec pipeline but will crash the regular text pipeline
            # max_vector_len = max(q_t.size(1) for q_t in batch_tensors)
            # min_vector_len = min(q_t.size(1) for q_t in batch_tensors)
            max_vector_len = max(q_t.size(0) for q_t in batch_tensors)
            min_vector_len = min(q_t.size(0) for q_t in batch_tensors)

            if max_vector_len != min_vector_len:
                # TODO: _pad_to_len move to utils
                from dpr.models.reader import _pad_to_len
                batch_tensors = [_pad_to_len(q.squeeze(0), 0, max_vector_len) for q in batch_tensors]

            q_ids_batch = torch.stack(batch_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            # if len(query_vectors) % 100 == 0:
                # logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    # logger.info("Total encoded queries tensor %s", query_tensor.size())
    assert query_tensor.size(0) == len(questions)
    return query_tensor

class DenseRetriever(object):
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.selector = None

    def generate_question_vectors(self, questions: List[str], query_token: str = None) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            query_token=query_token,
            selector=self.selector,
        )

class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, batch_size, tensorizer)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """


        time0 = time.time()
        print('start search knn !!!!!!!')
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        # self.index = None
        return results

    def search_with_L2(self, query_vectors: np.array, radius: int = 1):

        time0 = time.time()
        print('start range_search !!!!!!!')
        results = self.index.search_with_radius(query_vectors, radius) # range_search

        logger.info("index search time: %f sec.", time.time() - time0)
        # self.index = None
        return results


# =============================================================================================================================

@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):

    path = '/home/local/anaconda3/envs/paper/DPR/building_dataset/trec/'
    standard_distance = 'avg' # cfg.standard_distance
    distance_metric = 'cos_sim' # cfg.distance_metric

    batch_size = 4096
    
    cfg = setup_cfg_gpu(cfg)

    # saved_state = load_states_from_checkpoint(cfg.model_file)
    print('model loading --> this is bert-based')
    
    saved_state = load_states_from_checkpoint("/home/local/anaconda3/envs/paper/DPR/output/simcse_ckpt/dpr_biencoder.32")

    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    logger.info("Loading saved model state ...")
    encoder.load_state(saved_state, strict=False)

    encoder_path = cfg.encoder_path
    if encoder_path:
        logger.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    # encoder, _ = setup_for_distributed_mode(encoder, None, "cuda:1", cfg.n_gpu, cfg.local_rank, cfg.fp16)
    encoder.eval()

    model_to_load = get_model_obj(encoder)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    index_path = cfg.index_path # null

    # send data for indexing
    id_prefixes = []
    ctx_sources = []
    # for ctx_src in cfg.ctx_datatsets:
    for ctx_src in ['dpr_wiki']:
        ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
        id_prefixes.append(ctx_src.id_prefix)
        ctx_sources.append(ctx_src)
        logger.info("ctx_sources: %s", type(ctx_src))

    logger.info("id_prefixes per dataset: %s", id_prefixes)

    # simcse-based embedding
    ctx_files_patterns = ["/home/local/anaconda3/envs/paper/DPR/output/DenseEmbedding/SimCSE_embedding_0"]


    logger.info("ctx_files_patterns: %s", ctx_files_patterns)
    if ctx_files_patterns:
        assert len(ctx_files_patterns) == len(id_prefixes), "ctx len={} pref leb={}".format(
            len(ctx_files_patterns), len(id_prefixes)
        )
    else:
        assert (
            index_path or cfg.rpc_index_id
        ), "Either encoded_ctx_files or index_path pr rpc_index_id parameter should be set."

    input_paths = []
    path_id_prefixes = []
    for i, pattern in enumerate(ctx_files_patterns):
        pattern_files = glob.glob(pattern)
        pattern_id_prefix = id_prefixes[i]
        input_paths.extend(pattern_files)
        path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
        
    logger.info("Embeddings files id prefixes: %s", path_id_prefixes)
    logger.info("Reading all passages data from files: %s", input_paths)
    print('index start!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('\n')

    # enc -> get collection db for text
    all_passages = get_all_passages(ctx_sources)

    index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
    logger.info("Local Index class %s ", type(index))
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)

    print('retriever LocalFaissRetriever')
    retriever = LocalFaissRetriever(encoder, batch_size, tensorizer, index)
    retriever.index_encoded_data(input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes)

    if index_path:
        retriever.index.serialize(index_path)

    # load sst2 dataset for sampling queries from training dataset
    trec = load_dataset("trec")
    trec_train_df = pd.DataFrame(trec['train'])
    trec_train_df.drop(['label-fine'],axis=1,inplace=True)
    trec_train_df.columns = ['label','text']

    def construct_queries(data, num, seed):
        label_queries = []
        for label in np.unique(data['label']):
            label_query = []
            cond = data['label'] == label
            query = data[cond].sample(n=num, random_state=seed)
            label_query.append(query)
            label_queries.append(label_query)
        label_queries = np.array(label_queries)
        label_queries = label_queries.squeeze(1)
        return label_queries

    # query
    for query_number in [10]:

        for random_seed in [1234,5678,1004,9999,7777]:

            for subspace_size in [50]:

                print(f'{query_number} number of queries --> building start!!!!!!!!!!@!@!@!@@!@!@!!@')

                # query_path = f"/home/local/anaconda3/envs/paper/DPR/AGNews_query_ablation_output/query/query_num_{query_number}.csv"
                # total_queries = pd.read_csv(query_path)
                
                query_result = construct_queries(trec_train_df, query_number, random_seed)
                total_queries = pd.DataFrame(query_result.reshape(6*query_number,2), columns=['text','label'])
                total_queries.columns = ['label','text']
                for LABEL_trec in np.unique(total_queries['label']):
                    
                    questions = list(total_queries[total_queries['label'] == LABEL_trec]['text'])
                    logger.info("questions len %d", len(questions))

                    questions_tensor = retriever.generate_question_vectors(questions, query_token=None)

                    if distance_metric == 'cos_sim':
                        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                    
                        dist_list = []
                        for i, text_a in enumerate(questions_tensor):
                            for j, text_b in enumerate(questions_tensor):
                                if i != j and i < j:
                                    dist = cos(torch.tensor(text_a),torch.tensor(text_b))
                                    dist_list.append(dist)

                        if standard_distance == 'avg':
                            standard_dist = np.average(dist_list)

                        logger.info("standard_dist: %s", str(standard_dist))

                        # top_results_and_scores = retriever.get_top_docs(query_vectors = questions_tensor.numpy(), top_docs = 100)
                        top_results_and_scores = retriever.get_top_docs(query_vectors = questions_tensor.numpy(), top_docs = subspace_size)

                        retrieved_texts = []
                        for i, (content, score) in enumerate(top_results_and_scores):
                            print(f'{i}번째 retrieved texts')

                            query = questions_tensor[i]

                            tmp_text = []
                            for k in content:
                                text = all_passages[k]   
                                text_emb = retriever.generate_question_vectors(text, query_token=None) 
                                cos_sim = cos(torch.tensor(query),torch.tensor(text_emb[0]))
                                if  cos_sim > standard_dist:
                                    tmp_text.append(text[0])
                                else:
                                    continue

                            retrieved_texts.append(tmp_text)

                        retrieved_texts = sum(retrieved_texts,[])
                        print('positive samples retrieving clear!')

                        pos_dict = {'text':retrieved_texts}
                        pos_df = pd.DataFrame(pos_dict)
                        print(f'positive number: {pos_df.shape[0]}')


                        label_name = f"{LABEL_trec}th_label"

                        folder_path = path + f'QUERY_{query_number}_SS_SIZE_{subspace_size}/'
                        os.makedirs(folder_path, exist_ok=True)
                        pos_df.to_csv(folder_path + f"{label_name}_{random_seed}.csv")

if __name__ == "__main__":
    main()
    # HYDRA_FULL_ERROR=1
