from temporal_feature_processor.implementation.pandas.data.event import PandasEvent

from .base import PandasOperator


class PandasAssignOperator(PandasOperator):

  def __call__(self, event_1: PandasEvent, event_2: PandasEvent) -> PandasEvent:
    """Assign features to an event.
        Input event and features must have same index.
        Features cannot have more than one row for a single index + timestamp occurence.
        Output event will have same exact index and timestamps as input one.
        Assignment can be understood as a left join on the index and timestamp columns.

        Args:
            event (PandasEvent): event to assign the feature to.
            feature (PandasEvent): features to assign to the event.

        Returns:
            PandasEvent: a new event with the features assigned.
        """
    # assert indexes are the same
    if event_1.index.names != event_2.index.names:
      raise IndexError("Assign sequences must have the same index names.")

    # get index column names
    index, timestamp = self.split_index(event_1)

    # check there's no repeated timestamps index-wise in the assigned sequence
    if index:
      max_timestamps = (
          event_2.reset_index().groupby(index)[timestamp].value_counts().max())
    else:
      max_timestamps = event_2.reset_index()[timestamp].value_counts().max()

    if max_timestamps > 1:
      raise ValueError(
          "Cannot have repeated timestamps in assigned EventSequence.")

    # make assignment
    output = event_1.join(event_2, how="left", rsuffix="y")
    return output
