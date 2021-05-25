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

"""Export the source code comments into the API Reference website."""

import os

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api
import tensorflow_decision_forests as tfdf

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "/tmp/tfdf_api", "Where to output the docs")

flags.DEFINE_string("code_url_prefix", "", "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "/decision_forests/api_docs/python",
                    "Path prefix in the _toc.yaml")

flags.DEFINE_bool('gen_report', False,
                  ('Generate an API report containing the health of the'
                   'docstrings of the public API.'))

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow Decision Forests",
      py_modules=[("tfdf", tfdf)],  # ("tfdf.keras", tfdf.keras)
      base_dir=os.path.dirname(tfdf.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      gen_report=FLAGS.gen_report,
      callbacks=[public_api.explicit_package_contents_filter])
  doc_generator.build(FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
