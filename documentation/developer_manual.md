# Developer Manual

Internally, TD-DF relies on
[Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
(YDF). Depending on the change, reading YDF's user and developer manual might be
beneficial.

The library's dependency structure is organized in layers:

1.  Keras
2.  TensorFlow
3.  Python utility
4.  Yggdrasil

New logics should be implemented where relevant. When several layers are
possibly relevant, the most generic layer should be favored.

The directory structure of [TF-DF](directory_structure.md) and
[YDF](https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/directory_structure.md)
is a good start.
