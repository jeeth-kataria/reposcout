from dataclasses import dataclass


@dataclass(frozen=True)
class Parcel:
    postcode: str
    mass_grams: int
