# Directory Structure

The project is organized as follow:

```
├── examples: Collection of usage examples.
├── configure: Project configuration.
├── documentation: User and developer documentation. Contains the colabs.
├── component: Utilities.
│    ├── builder: Create models "by hand".
│    ├── inspector: Inspection of structure and meta-data of models.
│    ├── model_ploter: Plotting of model tree structure.
│    ├── inspector: Inspection of structure and meta-data of models.
│    └── py_tree: Representation of a decision tree as a python object.
├── third_party: Bazel configure for dependencies.
├── tools: Tools for the management of the project and code.
└── tensorflow_decision_forests: The library
    ├── keras: Keras logic. Depends on tensorflow logic.
    │   └── wrapper: Python code generator for Keras model.
    └── tensorflow: TensorFlow logic.
        └── ops: Custom C++ ops.
            ├── inference: ... for inference.
            └── training: ... for training.
```
