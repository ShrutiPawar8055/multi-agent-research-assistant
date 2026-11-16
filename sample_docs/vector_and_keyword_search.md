# Keyword and vector search

Keyword search ranks documents by the words they share with the query. BM25 is the best-known ranking function of this kind. It rewards rare query terms through inverse document frequency and dampens the effect of repeating a term many times, while normalising for document length so long passages are not unfairly favoured.

Vector search converts text into embeddings, which are lists of numbers where similar meanings sit close together. A query is embedded the same way and the nearest vectors are returned. Vector search finds paraphrases that share no words with the query, but it can miss exact identifiers such as error codes or product names.

Hybrid retrieval combines both signals. Many production systems run keyword and vector search together and merge the ranked lists, so exact matches and semantic matches both surface.
