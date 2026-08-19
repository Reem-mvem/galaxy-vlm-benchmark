"""TaskSpec: everything the generic runner needs to know about one task."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from vlm_eval.images import read_image_bytes
from vlm_eval.sampling import sample_all


@dataclass
class TaskSpec:
    """Declarative description of one evaluation task.

    The catalog DataFrame returned by ``load_catalog`` must contain at least
    ``id_str``, ``file_loc`` and ``label_name`` columns, where ``label_name``
    is one of ``class_names``.
    """

    name: str                      # slug, e.g. "is_lsb"
    label: str                     # pretty name for plot titles, e.g. "is-lsb"
    class_names: list              # index order must match dataset label ints
    prompt: str
    load_catalog: Callable[[], pd.DataFrame]
    parse_prediction: Callable[[str], Optional[str]]   # raw text -> class name or None
    dir: Path                      # task folder (anchor for default output paths)
    build_catalog: Optional[Callable[[], pd.DataFrame]] = None
    get_sample: Callable = sample_all
    prepare_image: Callable[[str], bytes] = read_image_bytes
    # extra per-row columns to carry into the results CSV (e.g. ring_fraction)
    record_extras: Callable[[pd.Series], dict] = field(default=lambda row: {})

    def load_or_build_catalog(self) -> pd.DataFrame:
        """Build the catalog on first use (downloads images), then reuse it."""
        try:
            return self.load_catalog()
        except FileNotFoundError:
            if self.build_catalog is None:
                raise
            return self.build_catalog()
