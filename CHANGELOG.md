# Changelog

## 0.1.4 - ????

### Features

-   Stop the training when interrupting a colab cell / typing ctrl-c.

### Bug fix

-   Fix failure when there are not input features.

## 0.1.3 - 2021-05-19

### Features

-   Register new inference engines.

## 0.1.2 - 2021-05-18

### Features

-   Inference engines: QuickScorer Extended and Pred

## 0.1.1 - 2021-05-17

### Features

-   Migration to TensorFlow 2.5.0.
-   By default, use a pre-compiled version of the OP wrappers.

### Bug fix

-   Add missing `plotter.js` from Pip package.
-   Use GitHub version of Yggdrasil Decision Forests by default.

## 0.1.0 - 2021-05-11

Initial Release of TensorFlow Decision Forests.

### Features

-   Random Forest learner.
-   Gradient Boosted Tree learner.
-   CART learner.
-   Model inspector: Inspect the internal model structure.
-   Model plotter: Plot decision trees.
-   Model builder: Create model "by hand".
