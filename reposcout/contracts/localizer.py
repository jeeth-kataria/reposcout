from typing import Protocol

from pydantic import Field

from .index import RepoIndex
from .models import ContractModel, Count, Instance, Localization, PositiveInt


class LocalizerConfig(ContractModel):
    max_llm_calls: Count = 12
    max_tool_calls: Count = 20
    max_explore_rounds: Count = 2
    max_output_tokens: PositiveInt = 2048
    timeout_s: float = Field(default=180, gt=0)
    top_k_files: PositiveInt = 10
    top_k_functions: PositiveInt = 10
    use_dense: bool = True
    use_graph_tools: bool = True
    use_verifier: bool = True


class Localizer(Protocol):
    def localize(
        self, instance: Instance, index: RepoIndex, config: LocalizerConfig
    ) -> Localization: ...
