from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from temporal_feature_processor.implementation.pandas.data.event import PandasEvent


class PandasOperator(ABC):
  """Base class to define an operator's interface."""

  @abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> PandasEvent:
    """Apply the operator to its inputs.

        Returns:
            PandasEvent: the output event of the operator.
        """

  def split_index(self, event: PandasEvent) -> Tuple[List[str], str]:
    """Split pandas' DataFrame index into (potentially) index columns and
        a timestamp column.
        Args:
            event (PandasEvent): input PandasEvent (pandas DataFrame).
        Returns:
            Tuple[List[str], str]: output index and timestamp names.
        """
    index_timestamp = event.index.names
    index = index_timestamp[:-1]
    timestamp = index_timestamp[-1]

    return index, timestamp
