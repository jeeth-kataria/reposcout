from .labels import shipping_label
from .parcel import Parcel


class DispatchDesk:
    @staticmethod
    async def manifest(parcels: list[Parcel]) -> list[str]:
        return [shipping_label(parcel) for parcel in parcels]
