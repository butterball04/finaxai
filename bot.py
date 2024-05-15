from llmsherpa.readers import LayoutPDFReader

import cohere
import uuid
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_url = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/pdf/S100TDDE.pdf?sv=2020-08-04&st=2024-05-15T15%3A31%3A36Z&se=2034-05-09T15%3A00%3A00Z&sr=b&sp=rl&sig=LEJGAhP6xnAyenLgeKRA8gs5F%2F2aLv2Hdcfse2xOJLU%3D"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)
doc = pdf_reader.read_pdf(pdf_url)

co_key = os.getenv("COHERE_API_KEY")
if not co_key:
    print("COHERE_API_KEY not set in environment variables")

co = cohere.Client(co_key)


class Vectorstore:

    def __init__(self, doc):
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 5
        self.load_and_chunk()
        self.embed()
        self.index()

    def load_and_chunk(self) -> None:
        """
        Loads the text from the source and chunks the content.
        """
        print("Loading documents...")

        for chunk in doc.chunks():
            self.docs.append(
                {
                    "title": "Mercari Quarterly Report",
                    "text": str(chunk.to_context_text()),
                    "url": pdf_url,
                }
            )

    def embed(self) -> None:
        """
        Embeds the document chunks.
        """
        print("Embedding document chunks...")

        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i: min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-multilingual-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the document chunks for efficient retrieval.
        """
        print("Indexing document chunks...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len,
                            ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(
            f"Indexing complete with {self.idx.get_current_count()} document chunks.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.

        """

        # Dense retrieval
        query_emb = co.embed(
            texts=[query], model="embed-multilingual-v3.0", input_type="search_query"
        ).embeddings

        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        # Reranking
        rank_fields = ["title", "text"]

        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-multilingual-v3.0",
            rank_fields=rank_fields
        )

        doc_ids_reranked = [doc_ids[result.index]
                            for result in rerank_results.results]

        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved
