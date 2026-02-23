"""
Early stopping utility expected by ORAT scripts.

ORAT uses:
    from earlystop import earlystop

So we provide a callable named `earlystop`.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class earlystop:
    patience: int = 10
    mode: str = "min"   # 'min' for loss, 'max' for accuracy
    delta: float = 0.0

    best: Optional[float] = None
    count: int = 0

    def __call__(self, metric: float) -> bool:
        # Returns True if training should stop
        if self.best is None:
            self.best = metric
            self.count = 0
            return False

        improved = (metric < self.best - self.delta) if self.mode == "min" else (metric > self.best + self.delta)

        if improved:
            self.best = metric
            self.count = 0
            return False

        self.count += 1
        return self.count >= self.patience
