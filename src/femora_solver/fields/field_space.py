from dataclasses import dataclass

@dataclass(frozen=True)
class FieldSpace:
    name: str      # e.g. "U"
    ncomp: int     # e.g. 3

    def __repr__(self):
        return f"FieldSpace({self.name!r}, {self.ncomp})"
