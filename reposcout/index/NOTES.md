# Index decisions and measurements

## Index construction

- tree-sitter 0.25.2 uses a Python grammar capsule from tree-sitter-python 0.25.0. The parser preserves intact definitions around error nodes. The AST-based call collector logs files it cannot parse. Such files still participate in retrieval. An empty Python file has a directory entry and zero symbols.
- Python source encoding declarations are honored before parsing. Decode failures, oversized files, non-Python files, submodules and symlinks contribute to skip counts. Directory listings expose indexed Python files and their ancestors.
- Names are scoped by their lexical parent. Repeated definitions such as overloads retain their individual retrieval spans. A symbol ID lookup returns the last definition in source order. Distinct module-level chunks share the module ID and have distinct spans.
- Calls retain the original caller, target expression, line and nullable resolved target in SQLite. Resolution checks same-class methods and known base classes for self/cls calls, then same-file names, explicit imports, and globally unique names for simple calls. Ambiguous targets stay unresolved. This heuristic does not perform dynamic dispatch, full Python MRO, runtime import evaluation, dependency injection or assignment-based inference. Imports from local scopes are approximated at file scope. Calls in function signatures and decorators are omitted.
- SQLite FTS5 BM25 indexes body, path and qualified name. Tokenization retains identifiers and dotted names and adds snake-case and camel-case components. Function, method and class bodies each receive a retrieval unit. Contiguous module code between definitions is split into chunks of at most 60 lines. Search snippets include an ellipsis when the displayed source is capped.
- The default dense model is the distilled RoBERTa Sentence Transformers encoder trained on CodeSearchNet. Its training pairs align text with code, its documented maximum input is 128 tokens, and it supports CPU batching. Embedding input includes the path, qualified name, signature, docstring and body from the source span. The tokenizer counts inputs exceeding the effective limit and the manifest records the number capped. Query caps are logged. Float16 vectors are persisted and expanded to normalized float32 in FAISS IndexFlatIP on first search. Sentence Transformers selects the available accelerator or CPU. [Model card](https://huggingface.co/flax-sentence-embeddings/st-codesearch-distilroberta-base)
- A SHA-256 of the file bytes identifies path-independent records within a versioned repository store. Git blob IDs allow cache hits without rereading the blob. Reused symbol paths, parents and test flags are rebound to the current path. The embedding cache keys the complete embedding text, model ID and token budget. Use immutable local model snapshots when comparing runs.
- Each repository has one advisory build lock. A complete SQLite snapshot and optional vector file are published by renaming a staged directory after the shared store commits. Lexical postings and graph edges are rebuilt for each new commit. Completed snapshots support concurrent read-only opens.

## Django benchmark

Measured on Apple arm64 with Python 3.12.2 using the locally cached Django Git objects. Both builds used lexical retrieval. Times include the SQLite writes. Each warm reopen loaded an existing snapshot.

| SWE-bench Lite base commit | Python files | Newly parsed blobs | Reused file entries | Parse and BM25 (s) | Graph (s) | Build (s) | Reopen (s) | Snapshot bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e7fd69d051eaa67cb17f172a39b57253e9cb831a | 2,534 | 1,995 | 539 | 12.81 | 0.88 | 13.78 | 0.033 | 76,185,600 |
| bceadd2788dc2dad53eba0caae172bd8522fd483 | 2,547 | 296 | 2,251 | 9.15 | 0.82 | 10.03 | 0.031 | 77,029,376 |

On the second commit 88.38% of file entries reused parsed content. Identical blobs within a revision also contribute to that count. The first snapshot's FTS postings occupied 14,082,048 bytes. Call graph tables and their indexes occupied 35,454,976 bytes. The shared content database occupied 19,218,432 bytes after the first build. Dense Django timing is unmeasured here.
