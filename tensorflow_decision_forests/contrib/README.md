# TensorFlow Decision Forests Contributions

The Contrib(ution) directory contains functionalities that are useful for TF-DF
users but which are not central to the TF-DF library. For example, it could
contain decision forest scientific paper implementations, utilities to interact
with other libraries (e.g. decision forests, IO), and model agnostic tools that
do not have another nice place to be.

Contrib libraries may not be as high-quality up-to-date as the rest of the TF-DF
core library, and updates may rely on contributors outside the TF-DF core team.

**Rules**

1.  There are no dependencies from the core library to the contrib libraries.
1.  The contrib libraries are not loaded automatically with the TF-DF library.
1.  A dependency cannot be registered in `configure/setup.py` if it is only used
    by a contrib library.
1.  If a contrib library becomes important enough (decided by the TF-DF core
    team), it can be moved to the `component` directory.
1.  The contrib directory is not a place to store usage examples.
