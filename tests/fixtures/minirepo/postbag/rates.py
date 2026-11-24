from .parcel import Parcel
from .zones import destination_zone


class RateCard:
    letter_limit_grams = 500

    def rate_band(self, grams: int) -> str:
        """Letters up to and including 500 g qualify for the letter rate."""
        if grams < self.letter_limit_grams:
            return "letter"
        return "parcel"

    def quote(self, parcel: Parcel) -> int:
        band = self.rate_band(parcel.mass_grams)
        base = 120 if band == "letter" else 360
        return base + (80 if destination_zone(parcel.postcode) == "national" else 0)
