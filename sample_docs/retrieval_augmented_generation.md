# Retrieval-augmented generation

Retrieval-augmented generation, often shortened to RAG, gives a language model access to external documents at answer time. The system first searches a collection for passages related to the question, then places those passages in the prompt so the model can ground its answer in them. This reduces hallucination because the model can quote or paraphrase real text instead of relying only on what it memorised during training.

A typical RAG pipeline has three stages. Documents are split into chunks and indexed ahead of time. At query time the retriever scores chunks against the question and returns the top few. Finally the generator writes an answer that cites those chunks. Chunk size matters: small chunks are precise but lose context, while large chunks keep context but dilute the relevance signal.

RAG is useful when knowledge changes often or is private, because updating the document collection is far cheaper than retraining a model.
