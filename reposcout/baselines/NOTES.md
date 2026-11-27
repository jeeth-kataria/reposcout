# Retrieval baseline decisions

- `RetrievalLocalizer` exposes retrieval depth and the RRF rank constant. The output config caps ranked files and functions independently. Sorting uses descending score, ascending path and start line, then symbol ID. The rationale identifies the retrieval method and depth.
- Fusion counts the first occurrence of a unit in each ranking. The key includes the path, optional symbol ID, and full inclusive span. Separate chunks can share a module ID. Raw lexical and dense scores are not assumed to share a scale.
- Every retrieved unit in a matching traceback file receives 0.025. The named function containing the frame line receives an additional 0.05. A frame can also introduce a previously unseen file. Exact paths, the longest repo-relative suffix of a longer frame path, and an unambiguous shortened relative frame path match. A renamed file or a different repository revision may prevent a match. These are fixed defaults before evaluation.
- Dense retrieval requires the index embedding extra and its model. A dense-only run fails explicitly if dense retrieval is disabled. The hybrid configurations support a lexical-only ablation.
