def destination_zone(postcode: str) -> str:
    def prefix(value: str) -> str:
        return value.strip().upper().split(" ", maxsplit=1)[0]

    return "local" if prefix(postcode).startswith("LS") else "national"
