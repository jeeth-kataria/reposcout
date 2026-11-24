# Postbag localization fixture

Issue text for the demo and integration check:

> A local letter weighing exactly 500 grams is quoted at 360p. The letter rate is 120p for all weights up to and including 500 grams.

Expected localization is `postbag/rates.py::RateCard.rate_band`. Keep the boundary bug in the fixture. Its own package tests are repository content for indexing. The top-level RepoScout suite uses this checkout as data.

The `minirepo_git` pytest fixture returns `root`, `base_commit`, and `head_commit`. The checkout starts at the second commit. Only the module-level service code in `postbag/express.py` changes. The pricing bug is present at both revisions.

The express service calls an inherited rate method. Both service modules define `label_prefix`, and the returns service decorates it with `@cache`. The incomplete customs module contains a valid function before a syntax error. The zero-byte empty module has no symbols.
