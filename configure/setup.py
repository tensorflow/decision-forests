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
import setuptools
from setuptools.command.install import install
from setuptools.dist import Distribution

_VERSION = "1.6.0rc0"

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "tensorflow>=2.13,<3",
    "six",
    "absl_py",
    "wheel",
    "wurlitzer",
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

try:
  from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

  class bdist_wheel(_bdist_wheel):

    def finalize_options(self):
      _bdist_wheel.finalize_options(self)
      self.root_is_pure = False

    def get_tag(self):
      python, abi, plat = _bdist_wheel.get_tag(self)
      if platform.system() == "Darwin":
        # Uncomment on of the lines below to adapt the platform string when
        # cross-compiling.
        # plat = "macosx_12_0_arm64"
        # plat = "macosx_10_15_x86_64"
        pass
      return python, abi, plat

except ImportError:
  bdist_wheel = None

setuptools.setup(
    cmdclass={
        "bdist_wheel": bdist_wheel,
        "install": InstallPlatlib,
    },
    name="tensorflow_decision_forests",
    version=_VERSION,
    author="Google Inc.",
    author_email="packages@tensorflow.org",
    description="Collection of training and inference decision forest algorithms.",
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
        "Programming Language :: Python :: 3.8",
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
    python_requires=">=3.8",
    license="Apache 2.0",
    keywords="tensorflow tensor machine learning decision forests random forest gradient boosted decision trees",
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=False,
)
