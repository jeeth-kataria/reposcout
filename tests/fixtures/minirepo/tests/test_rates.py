from postbag.parcel import Parcel
from postbag.rates import RateCard


def test_local_letter_price() -> None:
    assert RateCard().quote(Parcel("LS1 1AA", 400)) == 120
