from pathlib import Path

from reposcout.baselines import RetrievalLocalizer, reciprocal_rank_fusion, traceback_frames
from reposcout.contracts.localizer import LocalizerConfig
from reposcout.contracts.models import Instance, SearchHit, Span, Symbol
from reposcout.index.parser import extract_symbols


def hit(path: str, score: float) -> SearchHit:
    return SearchHit(
        symbol_id=None,
        span=Span(path=path, start_line=1, end_line=1),
        score=score,
        source="bm25",
        snippet="",
    )


def test_rrf_adds_reciprocal_ranks_and_breaks_ties() -> None:
    a, b, c = hit("a.py", 1), hit("b.py", 99), hit("c.py", 2)
    result = reciprocal_rank_fusion([[b, a, b], [c, a]])
    assert [h.span.path for h in result] == ["a.py", "b.py", "c.py"]
    assert result[0].score == 2 / 62 and result[1].score == 1 / 61


def test_traceback_uses_cited_file_and_function_without_seed_hit() -> None:
    issue = """Traceback (most recent call last):
  File "/srv/app/postbag/rates.py", line 10, in rate_band
    if grams < self.letter_limit_grams:
TypeError: '<' not supported
"""
    assert traceback_frames(issue) == [("/srv/app/postbag/rates.py", 10, "rate_band")]
    name = "postbag/rates.py"
    source = (Path(__file__).parents[1] / "fixtures" / "minirepo" / name).read_text()
    parsed = extract_symbols(source, name)

    class Index:
        def symbols(self) -> list[Symbol]:
            return parsed

        def get_symbol(self, symbol_id: str) -> Symbol | None:
            return next((s for s in parsed if s.id == symbol_id), None)

        def bm25(self, query: str, k: int) -> list[SearchHit]:
            return [hit("other.py", 1)]

        def dense(self, query: str, k: int) -> list[SearchHit]:
            return []

    case = Instance(instance_id="1", repo="demo/p", base_commit="abc", problem_statement=issue)
    guess = RetrievalLocalizer("hybrid_stacktrace").localize(case, Index(), LocalizerConfig())
    assert guess.files[0] == name
    assert guess.functions[0] == "postbag/rates.py::RateCard.rate_band"
    assert guess.stats.llm_calls == 0
    assert guess.candidates[0].citations[0].start_line == 8


def test_rrf_preserves_independent_chunks_of_the_same_module() -> None:
    first = hit("rates.py", 1)
    first.symbol_id = "rates.py::<module>"
    second = first.model_copy(update={"span": Span(path="rates.py", start_line=8, end_line=10)})
    result = reciprocal_rank_fusion([[first, second], [second]])
    assert [(h.span.start_line, h.score) for h in result] == [(8, 1 / 62 + 1 / 61), (1, 1 / 61)]


def test_trace_path_boost_increases_existing_nonframe_unit() -> None:
    from reposcout.baselines.retrieval import _boost_frames

    path = "postbag/rates.py"
    source = (Path(__file__).parents[1] / "fixtures" / "minirepo" / path).read_text()
    quote = next(s for s in extract_symbols(source, path) if s.name == "quote")

    class Index:
        def symbols(self) -> list[Symbol]:
            return extract_symbols(source, path)

    target = SearchHit(span=quote.span, symbol_id=quote.id, score=0.02, snippet="", source="rrf")
    frame = '  File "rates.py", line 10, in rate_band'
    boosted = _boost_frames([target, hit("other.py", 0.04)], frame, Index())
    assert boosted[0].symbol_id == path + "::RateCard.rate_band"
    assert boosted[1].symbol_id == quote.id and boosted[1].score == 0.045
