from .rates import RateCard

SERVICE_CODE = "X1"


def label_prefix() -> str:
    return SERVICE_CODE


class ExpressRateCard(RateCard):
    def describe(self, grams: int) -> str:
        return f"{label_prefix()} {self.rate_band(grams)}"
