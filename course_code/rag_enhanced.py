import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

from openai import OpenAI

#### CONFIG PARAMETERS ---

# Define the number of context chunks to consider for generating an answer.
NUM_CONTEXT_CHUNKS = 10
# Set the maximum length for each context chunk (in tokens).
MAX_CONTEXT_CHUNK_LENGTH = 256
# Set the stride for the sliding window (in tokens).
CHUNK_STRIDE = 128
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1

# VLLM Parameters
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85

# Sentence Transformer Parameters
SENTENCE_TRANSFORMER_BATCH_SIZE = 32

#### CONFIG PARAMETERS END---

class ChunkExtractor:
    def __init__(self):
        pass

    def _tokenize_text(self, text):
        """
        Tokenizes text into words. This is a simple whitespace tokenizer.
        """
        return text.split()

    def _reconstruct_text(self, tokens):
        """
        Reconstructs text from tokens.
        """
        return ' '.join(tokens)

    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts overlapping chunks from given HTML source using a sliding window approach.
        """
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]

        tokens = self._tokenize_text(text)
        chunks = []

        max_length = MAX_CONTEXT_CHUNK_LENGTH
        stride = CHUNK_STRIDE

        # Create overlapping chunks
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + max_length]
            chunk_text = self._reconstruct_text(chunk_tokens)
            chunks.append(chunk_text)

            if i + max_length >= len(tokens):
                break  # Stop if we've reached the end

        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all chunk extractions are complete
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

class EnhancedRAGModel:
    """
    An enhanced RAGModel that incorporates enhanced chunking, better embeddings, FAISS indexing,
    cross-encoder re-ranking, and refined prompt engineering.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()
        self.chunk_embeddings_index = None
        self.chunk_embeddings = None
        self.chunks = None
        self.chunk_interaction_ids = None

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # Initialize the model with vLLM server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # Initialize the model with vLLM offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",
                enforce_eager=True,
                max_model_len=4096
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a more powerful sentence transformer model
        self.sentence_model = SentenceTransformer(
            "all-mpnet-base-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

        # Load a cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def calculate_embeddings(self, texts):
        """
        Compute normalized embeddings for a list of texts using a sentence encoding model.
        """
        embeddings = self.sentence_model.encode(
            sentences=texts,
            normalize_embeddings=True,
            batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Extract chunks using improved chunking method
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate chunk embeddings if not already done
        if self.chunk_embeddings is None or not np.array_equal(self.chunks, chunks):
            self.chunks = chunks
            self.chunk_interaction_ids = chunk_interaction_ids
            self.chunk_embeddings = self.calculate_embeddings(chunks)
            self.index_chunk_embeddings()

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch using FAISS index
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx].reshape(1, -1)

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = self.chunk_interaction_ids == interaction_id
            relevant_chunk_indices = np.where(relevant_chunks_mask)[0]

            # Search in the FAISS index
            D, I = self.chunk_embeddings_index.search(query_embedding, NUM_CONTEXT_CHUNKS)
            retrieved_indices = I[0]

            # Map back to the original chunks
            retrieval_results = []
            for idx in retrieved_indices:
                if idx in relevant_chunk_indices:
                    retrieval_results.append(self.chunks[idx])

            # Re-rank using cross-encoder
            if retrieval_results:
                retrieval_results = self.rerank_chunks(query, retrieval_results)

            batch_retrieval_results.append(retrieval_results)

        # Prepare formatted prompts for the LLM
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)

        # Generate responses via vLLM
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,
                top_p=0.9,
                temperature=0.1,
                max_tokens=75,
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,
                    top_p=0.9,
                    temperature=0.1,
                    skip_special_tokens=True,
                    max_tokens=75,
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answers.append(response.outputs[0].text)

        return answers

    def index_chunk_embeddings(self):
        """
        Index chunk embeddings using FAISS for efficient retrieval.
        """
        embedding_dim = self.chunk_embeddings.shape[1]
        self.chunk_embeddings_index = faiss.IndexFlatIP(embedding_dim)
        self.chunk_embeddings_index.add(self.chunk_embeddings)

    def rerank_chunks(self, query, chunks):
        """
        Re-rank retrieved chunks using a cross-encoder.
        """
        cross_encoder_inputs = [[query, chunk] for chunk in chunks]
        scores = self.cross_encoder.predict(cross_encoder_inputs)
        reranked_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
        return reranked_chunks[:NUM_CONTEXT_CHUNKS]

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times, and retrieval results using the chat template of the model.
        """
        system_prompt = (
            """You are a highly intelligent and knowledgeable assistant specializing in providing accurate and concise answers. Your task is to answer questions based solely on the provided context. 
            - If the context contains sufficient information, answer the question directly and succinctly, no need to provide reasoning.
            - If the context lacks the necessary information, respond with "I don't know." Avoid guessing or fabricating answers.
            - Do not include any information that is not explicitly supported by the provided context.
            - Whenever possible, use the terminology and style from the references to ensure consistency and clarity.
            - Whenever the question is unreasonable or invalid, respond with "invalid question".

            Be professional, concise, and factual in your responses."""
        )
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""

            if retrieval_results:
                references += "Context:\n"
                for chunk in retrieval_results:
                    references += f"{chunk.strip()}\n"

            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

            user_message += f"{references}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts
