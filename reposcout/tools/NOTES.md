# Tool decisions

- The index reads regular Git blobs from the pinned commit and skips symlinks. Tool schemas validate file paths before requesting reads.
- Search displays up to three numbered source lines per hit and cites those displayed lines. Output identifies truncation.
- A run can attach a new list to `ToolLog` before its first tool call and restore the context token afterwards. Entries include raw arguments, latency, output size, truncation and error.
- Symbol detail previews the signature and opening docstring within the configured line budget. Long decorator blocks and multiline signatures can use the preview, in which case a subsequent range read can fetch the docstring.
