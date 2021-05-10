:: Copyright 2021 Google LLC.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     https://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

:: Compile and runs the unit tests.
set BAZEL=bazel-3.7.2-windows-x86_64.exe

:: TensorFlow bazelrc is required for the distributed compilation using
:: RBE i.e. a remove server (fast).
:: https://raw.githubusercontent.com/tensorflow/tensorflow/master/.bazelrc
SET TENSORFLOW_BAZELRC=tensorflow_bazelrc

%BAZEL% --bazelrc=%TENSORFLOW_BAZELRC% build^
  //tensorflow_decision_forests/...:all^
  --config=windows^
  --config=rbe_win^
  --config=rbe_win_py38^
  --config=tensorflow_testing_rbe_win^
  --flaky_test_attempts=1

:: Use --output_user_root to specify a quick-to-access location.

if %errorlevel% neq 0 exit /b %errorlevel%
