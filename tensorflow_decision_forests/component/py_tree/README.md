This directory contains a python implementation of decision trees, that can be
used to inspect and manipulate the decision forests trained with TF-DF using
python.

The internal and efficient decision forest representation can be converted back
and forth to this python representation for inspection and even arbitrary
manipulation.

Note: as one can expect this python representation is more explicit but much
less efficient than the Proto model (also referred as **core model**) used
internally by TF-DF. Complex algorithms should preferably be implemented in C++
on the Proto model.

A `Tree` is composed of a single `AbstractNode` called the *root*. This node and
its children recursively defines a decision tree.

If a node is a `LeafNode`, it contains an `AbstractValue` defining the
output/prediction/value of the node. Depending on the tree type, the
`AbstractValue` can be a `ProbabilityValue` or a `RegressionValue`.

If this node is `NonLeafNode`, it contains an `AbstractCondition`, two children
nodes, and optionally an `AbstractValue` for model interpretation.

Different implementations of `AbstractCondition` support different types of
conditions e.g. `NumericalHigherThanCondition`, `CategoricalIsInCondition`.

All objects can be printed using `str(...)`.
