# Directory Structure

The project is organised as follows:

```
├── configure: Project configuration.
├── documentation: User and developer documentation. Contains the colabs.
├── examples: Collection of usage examples.
├── tensorflow_decision_forests: The library
│   ├── component: Utilities.
│   │   ├── builder: Create models "by hand".
│   │   ├── inspector: Inspection of structure and meta-data of models.
│   │   ├── model_plotter: Plotting of model tree structure.
│   │   ├── inspector: Inspection of structure and meta-data of models.
│   │   ├── py_tree: Representation of a decision tree as a python object.
│   │   └── tuner: TF-DF's own hyper-parameter tuner.
│   ├── contrib: Additional functionality outside the project's main scope.
│   ├── keras: Keras logic. Depends on tensorflow logic.
│   │   └── wrapper: Python code generator for Keras models.
│   │── tensorflow: TensorFlow logic.
│   │   └── ops: Custom C++ ops.
│   │       ├── inference: ... for inference.
│   │       └── training: ... for training.
│   └── test_data: Datasets for unit tests and benchmarks.
├── third_party: Bazel configuration for dependencies.
└── tools: Tools for the management of the project and code.
```
