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

"""Setup file for pip's build.

This file is used by tools/build_pip_package.sh.
"""

import platform
import sys
import setuptools
from setuptools.command.install import install
from setuptools.dist import Distribution

_VERSION = "1.10.0"

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "tensorflow==2.17.0",
    "six",
    "absl_py",
    "wheel",
    "wurlitzer",
    "tf_keras~=2.17",
    "ydf",
]


class InstallPlatlib(install):

  def finalize_options(self):
    install.finalize_options(self)
    if self.distribution.has_ext_modules():
      self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False


if "bdist_wheel" in sys.argv:
  if "--plat-name" not in sys.argv:
    if platform.system() == "Darwin":
      idx = sys.argv.index("bdist_wheel") + 1
      sys.argv.insert(idx, "--plat-name")
      if platform.processor() == "arm":
        sys.argv.insert(idx + 1, "macosx_12_0_arm64")
      elif platform.processor() == "i386":
        sys.argv.insert(idx + 1, "macosx_10_15_x86_64")
      else:
        raise ValueError(f"Unknown processor {platform.processor()}")
    else:
      print("Not on MacOS")
  else:
    print("PLAT-NAME Supplied")
else:
  print("NO BDIST_WHEEL")

setuptools.setup(
    cmdclass={
        "install": InstallPlatlib,
    },
    name="tensorflow_decision_forests",
    version=_VERSION,
    author="Google Inc.",
    author_email="decision-forests-contact@google.com",
    description=(
        "Collection of training and inference decision forest algorithms."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorflow/decision-forests",
    project_urls={
        "Bug Tracker": "https://github.com/tensorflow/decision-forests/issues",
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    distclass=BinaryDistribution,
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    license="Apache 2.0",
    keywords=(
        "tensorflow tensor machine learning decision forests random forest"
        " gradient boosted decision trees"
    ),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=False,
)
