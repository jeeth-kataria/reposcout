from .parcel import Parcel
from .rates import RateCard


def shipping_label(parcel: Parcel) -> str:
    amount = RateCard().quote(parcel)
    return f"{parcel.postcode.upper()} | {amount}p"
