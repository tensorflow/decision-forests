# Ways to consume text with Tensorflow Decision Forest models

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'arvnd' reviewed: '2021-07-02' }
*-->

This is a documentation Markdown page. For more information, see the Markdown Reference
(go/documentation-reference) and the documentation Style Guide (go/documentation-style).

[TOC]

#### Background

Decision forest learning algorithms work differently than gradient descent based
models like neural networks or linear predictors. These differences manifest
themselves across a variety of modeling decisions, but are especially pronounced
when a dataset contains variable length categorical features, like tokenized
text features, which tend to require architecture-specific feature engineering.
This guide outlines the tradeoffs between different feature engineering
strategies for text features in decision forest models.

In the following sections, we will refer to a dataset with these features, and
assume we are predicting whether a user is going to purchase a specific product:

<table>
  <tr>
   <td><strong>Feature</strong>
   </td>
   <td>User ID
   </td>
   <td>Prior Purchases
   </td>
   <td>Product Title
   </td>
   <td>Product Description
   </td>
  </tr>
  <tr>
   <td><strong>Example data</strong>
   </td>
   <td>1234
   </td>
   <td>[“TV”, “Vacuum”]
   </td>
   <td>“WiFi Router”
   </td>
   <td>“This router is …”
   </td>
  </tr>
</table>

\
In this example, “Prior Purchases” is a sparse text feature (or a set of
categorical items). “Product Title” is as well, but is not tokenized in this
example. “Product Description” is a natural language feature, which has
different properties than the other features, because we expect the vocabulary
to be large (or unbounded), for word-order to matter, and to have other semantic
and lexical properties inherent to the language. The strategies we describe
below are appropriate for all of these features, but will have different
tradeoffs for each one.

#### Quick Reference

The best solution, if training and inference cost is not a concern, is to use
both categorical-sets and pre-trained embeddings for each text feature, since
they have complementary strengths and weaknesses. We recommend this unless one
of the constraints mentioned below are present.

<table>
  <tr>
   <td>
   </td>
   <td><strong>Inference Speed</strong>
   </td>
   <td><strong>Training Speed</strong>
   </td>
   <td><strong>Ability to memorize token &lt;> label relationships</strong>
   </td>
   <td><strong>Generalization</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Multiple Categoricals</strong>
   </td>
   <td>Fastest (++)
   </td>
   <td>Fastest (++)
   </td>
   <td>Limited
   </td>
   <td>Limited (+)
   </td>
  </tr>
  <tr>
   <td><strong>multi-hot</strong>
   </td>
   <td>Fast (+)
   </td>
   <td>Fast (assuming relatively small vocab size) (++)
   </td>
   <td>Good
   </td>
   <td>Limited (+)
   </td>
  </tr>
  <tr>
   <td><strong>Categorical-sets</strong>
   </td>
   <td><strong>Fastest (+++)</strong>
   </td>
   <td>Slower (+)
   </td>
   <td><strong>Best</strong>
   </td>
   <td>Limited (++)
   </td>
  </tr>
  <tr>
   <td><strong>embedding</strong>
   </td>
   <td>Slowest (assuming non-trivial encoder ops, like matrix multiplications) (+ to +++)
   </td>
   <td><strong>Fastest (assuming vocab size >> embedding dimension) (+++)</strong>
   </td>
   <td>Bad
   </td>
   <td><strong>Good (++ to +++)</strong>
   </td>
  </tr>
</table>

#### N-Grams

N-grams (e.g. {"the", "cat", "is", "blue"} -> {"&lt;start> the", "the cat", "cat
is", "is blue", "blue &lt;end>"}) can be beneficial in many cases, and capture
local lexical information. They are supported in all of the methods below, but
come at the cost of a dramatically larger vocabulary size, which can make them
impractical due to training cost.

### Discouraged Strategies

#### One-hot / Multi-hot encoding / Bag of Words

One-hot encoding is a classic strategy for densifying sparse text. Here we
assume an extension where a sparse text feature is represented by either a
multi-hot (1s for all contained tokens) or count-based vectorization (the count
for each token in the vocabulary).

For example, if the vocabulary is 4 items, and indexed like [“TV”, “Vacuum”,
“Wifi”, “Router”], the feature “Prior Purchases” would be a dense vector &lt;1,
1, 0, 0>. If counts were taken into account and the feature was [“TV”, “TV”,
“Vacuum”], it would be &lt;2, 1, 0, 0>.

**Pros**

*   Since decision forest splits are learned on individual features, this is
    less expensive at training time than categorical sets.
*   Unlike the former, does not need any clipping or padding, and tokens have
    the same semantics across examples (i.e. “TV” will be constant across splits
    regardless of position).

**Cons**

*   This strategy often leads to highly unbalanced and sparse splits, which can
    make DF learning algorithms either slower to converge or subpar. This is
    because:
    *   More splits are needed to learn the same information
    *   Highly sparse trees generalize worse than balanced trees, usually
        resulting in a less accurate model.
*   Does not take into account positional information. This may hurt performance
    for natural language features.
*   Learning numerical splits on categorical data is sub-optimal; there are
    optimizations for finding categorical splits that are not leveraged here.
*   The training computational complexity scales linearly with the number of
    vocabulary items (which will each be consumed as a numerical feature). In
    practice, unless the dataset is very small (in which case large vocabularies
    may encourage overfitting), this makes vocabularies of > 5k items very slow
    to train.
*   The training memory consumption will be 1 byte (for one-hot) or 4 bytes (for
    counts) per vocabulary item per example, since at indexing time, the data
    will be stored as a dense version of the sparse data. This can grow
    prohibitively large for larger vocabularies and datasets.

#### Multiple Categorical Features with a fixed length

Since categorical features can be efficiently learned by decision forest
algorithms, one natural way to consume sparse tokens is to pad / clip such that
there are a fixed number of input tokens per example, and each token position is
a separate and independent feature. In the example above, if “Prior Purchases”
has at most 5 tokens, we can create features f1...f5 representing tokens 1-5,
and discard any tokens > 5, and add missing values for examples where there are
&lt; 5.

**Pros**

*   This is efficient to train.
*   This may not hurt model quality if there is a low variance in the number of
    tokens per example, and the tokens are independent.
*   This may capture additional semantics (like purchase order in the example)
    more naturally than other methods.

**Cons**

*   Adds semantics onto “missing” padded tokens that will serve as noise to the
    model. This will be especially pronounced if there is a large variance in
    the number of tokens per example, which may happen for example with the
    “Product Description” feature.
*   The learned trees will be highly sensitive to ordering, i.e. if the feature
    is [“A”, “B”] the prediction will be different than the prediction for [“B”,
    “A”], and if the latter was never seen in the data, the model will be unable
    to generalize from the former. In general, this will require much more data
    to learn position invariance.
*   By default, each token will be represented by a feature with a different
    vocabulary. Even if you force the implementation to consider the same set of
    vocabulary items per feature, f1=”TV” will be a different vocabulary item
    than f2=”TV.” This means the algorithm will be less efficient in learning
    the relationship between the token “TV” and the label -- it will have to
    learn it separately for each token position.

### Better Strategies

#### Categorical Sets

Categorical Sets (https://arxiv.org/pdf/2009.09991.pdf) are TF-DFs default
feature representation for sparse text. A categorical set is effectively a bag
of words, ignoring duplicates and ordering. For example, the feature “The TV is
the best” would be represented by the categorical set {“best”, “is”, “the”,
“TV}.

The native categorical set splitter, according to benchmarks on a variety of
tasks (see paper), usually outperforms multi-hot and fixed-length categorical
features. In theory, both categorical set splits and boolean splits on one-hot
encoded features can learn the same information. However, take the following
example, where the tree is trying to learn the following function:

```
if description contains “high” AND “speed” AND “gaming”:
  return True
```

In this case, the native categorical set splitter will learn 1 split, where
{“high”, “speed”, “gaming”} => True.

A one hot representation would require 3 splits, on “high”, “split”, and
“gaming,” and would need to find reasonable leaf nodes for all possible
disjunctions of those categories (i.e. “high” and not “speed”). In practice,
one-hot encoding leads to highly unbalanced trees that cannot be optimized
efficiently by the best performing decision forest learning algorithms.

Pros

*   Best at learning bag-of-words information for decision forests.
*   Highly efficient to serve (can be served with QuickScorer, which can serve
    large trees in up to sub-microsecond-per-example time). The serving time
    complexity is linear in the number of items in each example’s categorical
    set, which in practice is much smaller than the vocabulary size.
*   Optimizes a single vocabulary per feature, so semantics are shared.

Cons

*   The cost of training a categorical set split scales with num\_examples \*
    vocab size, so similar to the one-hot algorithm, the trainable vocabulary
    size can be fairly small (N thousand) in practical settings. Note that this
    training speed can be improved by adjusting the sampling fraction of the
    greedy algorithm, but it may achieve sub-optimal quality.

#### Embeddings

Neural Networks have displayed state of the art performance on a variety of NLP
tasks, and pre-trained embeddings consumed as numerical features empirically
also work well with decision forest algorithms, despite the features being used
very differently internally. Note that here we refer to “embedding” as any
neural network encoding, e.g. the output of transformer / convolutional /
recurrent layers.

Using pre-trained embeddings works well with neural networks in part because the
initialization of a vector space where similar tokens or sentences are close in
euclidean space has shown to transfer well across NLP tasks, and the gradients
from that initialization are smaller and faster to converge than a fully random
initialization. However, decision trees use embeddings as individual numeric
features, and learn axis-aligned partitions of those individual features[^1].
This means it is near impossible to utilize the same semantic information -- a
dot product or a matrix multiplication, for example, cannot be represented with
a set of axis-aligned splits. Furthermore, unlike neural networks, which can
update the embeddings through gradient descent during training, the default
decision forest learning algorithms are non-differentiable, meaning that the
embeddings must stay frozen. Note that there is work
(https://arxiv.org/pdf/2007.14761.pdf, for example) on differentiable decision
forests. However, perhaps in part because in practice not all the bits of
information in an embedding are actually utilized, even by neural networks, this
strategy still works well with decision forests.

**Pros:**

*   Can deal with much larger vocabulary sizes -- since an embedding is
    effectively a densification into a small number of embedding dimensions, it
    is unlikely that the number of input features to the decision forest
    increases dramatically.
*   Can generalize better, in theory, since similar embeddings can share sets of
    partitions. Note that a big caveat here is that, as mentioned above, basis
    transformations or rotations in vector space can have two similar embeddings
    be completely different in the axis-aligned partitioned space for decision
    forests.
*   Can naturally encapsulate recurrence / word order, for example if the
    encoder contains convolutions, attention, or an RNN.
*   Can leverage information from another dataset (pre-training for transfer
    learning).

**Cons**

*   Not good at memorizing information -- the splits can cause fuzzy
    classifications or high sensitivity to phrasing (i.e. “the router is great”
    vs “a great router”) will produce different embeddings, which may be close
    in euclidean space but not necessarily have similar numerical features.
*   Slowest to serve, because the full encoder forward pass needs to be done at
    inference time. The actual latency is highly dependent on the architecture
    that produced the embeddings; i.e., a transformer encoder will typically be
    much slower than a raw embedding table lookup with mean-pooling.

<!-- Footnotes themselves at the bottom. -->

## Notes

[^1]: Enabling oblique splits can allow learning non-axis aligned information,
    but it will still be on a dimension-by-dimension basis.
