# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example of Contrib.

Usage example:

```python
import tensorflow_decision_forests as tfdf
from tensorflow_decision_forests.contrib import example_of_contrib

print(example_of_contrib.my_function())
```
"""

from tensorflow_decision_forests.contrib.example_of_contrib import example_of_contrib as lib

my_function = lib.my_function
