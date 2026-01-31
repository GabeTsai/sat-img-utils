from __future__ import annotations
from dataclasses import dataclass, field, asdict, is_dataclass
from collections.abc import Mapping, Iterator
from typing import Any

from sat_img_utils.pipelines.config import RESERVED

class NS:
    """
    Namespace wrapper around dict for attribute-style access.
    """
    def __init__(self, d):
        object.__setattr__(self, "_d", dict(d))

    def __getattr__(self, k):
        v = self._d[k]
        return NS(v) if isinstance(v, dict) else v
    
    def __setattr__(self, k, v):
        self._d[k] = v

@dataclass(frozen=True)
class Context(Mapping[str, Any]):
    """
    General context object for pipelines. 
    Cfg contains invariant configuration parameters.
    Extra contains any additional context parameters. 
    Each key in extra corresponds to a pipeline component (ie a single filter or transform) 
    that may contain its own parameters.
    Extras override cfg on name collisions.
    """
    cfg: Any
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self.extra:
            v = self.extra[name]
            return NS(v) if isinstance(v, dict) else v
        return getattr(self.cfg, name)  # raises AttributeError if missing

    def __getitem__(self, key: str) -> Any:
        if key in self.extra:
            v = self.extra[key]
            return NS(v) if isinstance(v, dict) else v
        if is_dataclass(self.cfg):
            d = asdict(self.cfg)
            if key in d:
                return d[key]
        return getattr(self.cfg, key)  #

    def __iter__(self) -> Iterator[str]:
        keys = set(self.extra.keys())
        if is_dataclass(self.cfg):
            keys |= set(asdict(self.cfg).keys())
        else:
            #cfg attributes aren't enumerable; only extras enumerable
            pass
        return iter(keys)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def merged_dict(self) -> dict[str, Any]:
        base = asdict(self.cfg) if is_dataclass(self.cfg) else {}
        return {**base, **dict(self.extra)}
    
    def _validate_extra(self, extra: Mapping[str, Any]) -> None:  
        for k, v in extra.items():
            if k in RESERVED:
                if not isinstance(v, Mapping):
                    raise TypeError(f"extra['{k}'] must be a mapping/dict, got {type(v)}")
                continue

            if not isinstance(v, Mapping):
                raise TypeError(
                    f"extra['{k}'] must be a dict of component params; "
                    f"top-level scalar keys are not allowed (got {type(v)})"
                )
    
    def __post_init__(self) -> None:
        self._validate_extra(self.extra)

