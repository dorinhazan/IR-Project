import numpy as np
import math
from collections import defaultdict


class BM25:
    def __init__(self, index, index_DL ,k1=1.5, b=0.75):
        """
        Initializes the BM25Optimized class with an index, optional storage location, and BM25 parameters.
        :param index: The inverted index object.
        :param storage_location: The base directory or bucket name for cloud storage.
        :param k1: BM25 k1 parameter.
        :param b: BM25 b parameter.
        """
        self.index = index
        self.k1 = k1
        self.b = b
        self.N = len(index_DL)
        self.AVGDL = np.mean(list(index_DL.values()))

    def _calc_idf(self, list_of_tokens):
        """
        Calculates the inverse document frequency (IDF) for a list of tokens.
        :param list_of_tokens: A list of tokens (words) for which to calculate IDF.
        :return: A dictionary mapping tokens to their IDF values.
        """
        idf = {}
        for term in np.unique(list_of_tokens):
            if term in self.index.df:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
        return idf

    def _score(self, query, doc_id, term_frequencies, idf):
        """
        Calculates the BM25 score for a single document and query.
        :param query: The query terms.
        :param doc_id: The document ID.
        :param term_frequencies: A dictionary of term frequencies in the document.
        :param idf: A dictionary of IDF values for terms.
        :return: The BM25 score for the document.
        """
        score = 0.0
        doc_len = self.index.DL[doc_id]
        for term in query:
            if term in term_frequencies:
                freq = next((tf for doc, tf in term_frequencies[term] if doc == doc_id), 0)
                numerator = idf.get(term, 0) * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + (self.b * doc_len / self.AVGDL))
                score += numerator / denominator if denominator != 0 else 0
        return score

    def search(self, query, N=100):
        """
        Performs a BM25 search for a query, returning the top N documents ranked by their score.
        :param query: The query terms.
        :param N: The number of top documents to return.
        :return: A list of tuples (doc_id, score) for the top N documents.
        """
        idf = self._calc_idf(query)
        term_frequencies = {}
        for term in query:
            if term in self.index.df:
                term_frequencies[term] = self.index.read_a_posting_list("",term, "nettaya-315443382")

        candidates = self.get_candidate_documents(query)
        scores = [(doc_id, self._score(query, doc_id, term_frequencies, idf)) for doc_id in candidates]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:N]

    def get_candidate_documents(self, query):
        """
        Retrieves candidate documents that contain at least one term from the query.
        :param query: The query terms.
        :return: A set of document IDs that are candidates for retrieval.
        """
        candidates = set()
        for term in np.unique(query):
            if term in self.index.df:
                postings = self.index.read_a_posting_list("",term, "nettaya-315443382")
                candidates.update([doc_id for doc_id, _ in postings])
        return candidates
