/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_decision_forests/keras/wrapper/wrapper.h"

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace tensorflow {
namespace decision_forests {

namespace ydf = yggdrasil_decision_forests;

// Gets the number of prefix spaces.
int NumLeadingSpaces(const absl::string_view text) {
  auto char_it = text.begin();
  while (char_it != text.end() && *char_it == ' ') {
    char_it++;
  }
  return std::distance(text.begin(), char_it);
}

// Converts a learner name into a python class name.
// e.g. RANDOM_FOREST -> RandomForestModel
std::string LearnerKeyToClassName(absl::string_view key) {
  std::string value(key);
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (it == value.begin() || !absl::ascii_isalpha(*(it - 1))) {
      *it = absl::ascii_toupper(*it);
    } else {
      *it = absl::ascii_tolower(*it);
    }
  }
  return absl::StrCat(absl::StrReplaceAll(value, {{"_", ""}}), "Model");
}

// Converts a learner name into a nice name.
// e.g. "RANDOM_FOREST" -> "Random Forest"
std::string LearnerKeyToNiceLearnerName(absl::string_view key) {
  std::string value(key);
  for (auto it = value.begin(); it != value.end(); ++it) {
    if (it == value.begin() || !absl::ascii_isalpha(*(it - 1))) {
      *it = absl::ascii_toupper(*it);
    } else {
      *it = absl::ascii_tolower(*it);
    }
  }
  return absl::StrReplaceAll(value, {{"_", " "}});
}

// Converts a floating point value to its python representation.
std::string PythonFloat(const float value) {
  std::string str_value;
  absl::StrAppendFormat(&str_value, "%g", value);
  // Check if the number is finite and written in decimal notation.
  if (std::isfinite(value) && !absl::StrContains(str_value, "+")) {
    // Make sure the value is a python floating point number.
    if (!absl::StrContains(str_value, ".")) {
      absl::StrAppend(&str_value, ".0");
    }
  }
  return str_value;
}

// Generates the python documentation and python object for the pre-defined
// hyper-parameters.
ydf::utils::StatusOr<std::pair<std::string, std::string>>
BuildPredefinedHyperParameter(const ydf::model::AbstractLearner* learner) {
  // Documentation about the list of template hyper-parameters.
  std::string predefined_hp_doc;
  // Python list of template hyper-parameters.
  std::string predefined_hp_list = "[";

  const auto predefined_hyper_parameter_sets =
      learner->PredefinedHyperParameters();
  for (const auto& predefined : predefined_hyper_parameter_sets) {
    absl::SubstituteAndAppend(
        &predefined_hp_doc,
        "        - $0@v$1: $2 The parameters are: ", predefined.name(),
        predefined.version(), predefined.description());
    absl::SubstituteAndAppend(
        &predefined_hp_list,
        "core.HyperParameterTemplate(name=\"$0\", "
        "version=$1, description=\"$2\", parameters={",
        predefined.name(), predefined.version(),
        absl::StrReplaceAll(predefined.description(), {{"\"", "\\\""}}));

    // Iterate over the individual parameters.
    bool first_field = true;
    for (const auto& field : predefined.parameters().fields()) {
      if (first_field) {
        first_field = false;
      } else {
        absl::StrAppend(&predefined_hp_doc, ", ");
        absl::StrAppend(&predefined_hp_list, ", ");
      }
      absl::StrAppend(&predefined_hp_doc, field.name(), "=");
      absl::StrAppend(&predefined_hp_list, "\"", field.name(), "\" :");
      switch (field.value().Type_case()) {
        case ydf::model::proto::GenericHyperParameters_Value::TYPE_NOT_SET:
          return absl::InternalError("Non configured value");
          break;
        case ydf::model::proto::GenericHyperParameters_Value::kCategorical: {
          std::string value = field.value().categorical();
          if (value == "true" || value == "false") {
            value = (value == "true") ? "True" : "False";
          } else {
            value = absl::StrCat("\"", value, "\"");
          }
          absl::StrAppend(&predefined_hp_doc, value);
          absl::StrAppend(&predefined_hp_list, value);
        } break;
        case ydf::model::proto::GenericHyperParameters_Value::kInteger:
          absl::StrAppend(&predefined_hp_doc, field.value().integer());
          absl::StrAppend(&predefined_hp_list, field.value().integer());
          break;
        case ydf::model::proto::GenericHyperParameters_Value::kReal:
          absl::StrAppend(&predefined_hp_doc,
                          PythonFloat(field.value().real()));
          absl::StrAppend(&predefined_hp_list,
                          PythonFloat(field.value().real()));
          break;
        case ydf::model::proto::GenericHyperParameters_Value::kCategoricalList:
          absl::StrAppend(
              &predefined_hp_doc, "[",
              absl::StrJoin(field.value().categorical_list().values(), ","),
              "]");
          absl::StrAppend(
              &predefined_hp_list, "[",
              absl::StrJoin(field.value().categorical_list().values(), ","),
              "]");
          break;
      }
    }
    absl::SubstituteAndAppend(&predefined_hp_doc, ".\n");
    absl::SubstituteAndAppend(&predefined_hp_list, "}),");
  }
  absl::StrAppend(&predefined_hp_list, "]");
  return std::pair<std::string, std::string>(predefined_hp_doc,
                                             predefined_hp_list);
}

// Formats some documentation.
//
// Args:
//   raw: Raw documentation.
//   leading_spaces_first_line: Left margin on the first line.
//   leading_spaces_next_lines: Left margin on the next lines.
//
std::string FormatDocumentation(const absl::string_view raw,
                                const int leading_spaces_first_line,
                                const int leading_spaces_next_lines) {
  // Sanitize documentation.
  std::string raw_sanitized = absl::StrReplaceAll(raw, {{"\\", "\\\\"}});

  // Extract the lines of text.
  const std::vector<std::string> lines = absl::StrSplit(raw_sanitized, '\n');
  std::string formatted;

  for (int line_idx = 0; line_idx < lines.size(); line_idx++) {
    const auto& line = lines[line_idx];

    // Leading spaces of the current line.
    const int user_leading_spaces = NumLeadingSpaces(line);

    const auto leading_spaces =
        (line_idx == 0) ? leading_spaces_first_line : leading_spaces_next_lines;

    int written_length = leading_spaces + user_leading_spaces;
    absl::StrAppend(&formatted, std::string(written_length, ' '));

    const std::vector<std::string> tokens = absl::StrSplit(line, ' ');
    for (int token_idx = 0; token_idx < tokens.size(); token_idx++) {
      const auto& token = tokens[token_idx];
      if (token_idx > 0) {
        absl::StrAppend(&formatted, " ");
      }
      absl::StrAppend(&formatted, token);
      written_length += token.size() + 1;
    }

    // Tailing line return.
    absl::StrAppend(&formatted, "\n");
  }
  return formatted;
}

ydf::utils::StatusOr<std::string> GenKerasPythonWrapper() {
  const auto prefix = "";

  std::string imports = absl::Substitute(R"(
from $0tensorflow_decision_forests.keras import core
from $0yggdrasil_decision_forests.model import abstract_model_pb2  # pylint: disable=unused-import
)",
                                         prefix);

  std::string wrapper =
      absl::Substitute(R"(r"""Wrapper around each learning algorithm.

This file is generated automatically by running the following commands:
  bazel build -c opt //third_party/tensorflow_decision_forests/keras:wrappers
  bazel-bin/third_party/tensorflow_decision_forests/keras/wrappers_wrapper_main\
    > third_party/tensorflow_decision_forests/keras/wrappers_pre_generated.py

Please don't change this file directly. Instead, changes the source. The
documentation source is contained in the "GetGenericHyperParameterSpecification"
method of each learner e.g. GetGenericHyperParameterSpecification in
learner/gradient_boosted_trees/gradient_boosted_trees.cc contains the
documentation (and meta-data) used to generate this file.
"""

from typing import Optional, List, Set
import tensorflow as tf
$0
TaskType = "abstract_model_pb2.Task"  # pylint: disable=invalid-name
AdvancedArguments = core.AdvancedArguments

)",
                       imports);

  for (const auto& learner_key : ydf::model::AllRegisteredLearners()) {
    const auto class_name = LearnerKeyToClassName(learner_key);

    // Get a learner instance.
    std::unique_ptr<ydf::model::AbstractLearner> learner;
    ydf::model::proto::TrainingConfig train_config;
    train_config.set_learner(learner_key);
    train_config.set_label("my_label");
    RETURN_IF_ERROR(GetLearner(train_config, &learner));
    ASSIGN_OR_RETURN(const auto specifications,
                     learner->GetGenericHyperParameterSpecification());

    // Python documentation.
    std::string fields_documentation;
    // Constructor arguments.
    std::string fields_constructor;
    // Use of constructor arguments the parameter dictionary.
    std::string fields_dict;

    // Sort the fields alphabetically.
    std::vector<std::string> field_names;
    field_names.reserve(specifications.fields_size());
    for (const auto& field : specifications.fields()) {
      field_names.push_back(field.first);
    }
    std::sort(field_names.begin(), field_names.end());

    for (const auto& field_name : field_names) {
      const auto& field_def = specifications.fields().find(field_name)->second;

      if (field_def.documentation().deprecated()) {
        // Deprecated fields are not exported.
        continue;
      }

      // Constructor argument.
      if (!fields_constructor.empty()) {
        absl::StrAppend(&fields_constructor, ",\n");
      }
      // Type of the attribute.
      std::string attr_py_type;
      // Default value of the attribute.
      std::string attr_py_default_value;

      if (ydf::utils::HyperParameterIsBoolean(field_def)) {
        // Boolean values are stored as categorical.
        attr_py_type = "bool";
        attr_py_default_value =
            (field_def.categorical().default_value() == "true") ? "True"
                                                                : "False";
      } else {
        switch (field_def.Type_case()) {
          case ydf::model::proto::GenericHyperParameterSpecification::Value::
              kCategorical: {
            attr_py_type = "str";
            absl::SubstituteAndAppend(&attr_py_default_value, "\"$0\"",
                                      field_def.categorical().default_value());
          } break;
          case ydf::model::proto::GenericHyperParameterSpecification::Value::
              kInteger:
            attr_py_type = "int";
            absl::StrAppend(&attr_py_default_value,
                            field_def.integer().default_value());
            break;
          case ydf::model::proto::GenericHyperParameterSpecification::Value::
              kReal:
            attr_py_type = "float";
            absl::StrAppend(&attr_py_default_value,
                            PythonFloat(field_def.real().default_value()));
            break;
          case ydf::model::proto::GenericHyperParameterSpecification::Value::
              kCategoricalList:
            attr_py_type = "List[str]";
            attr_py_default_value = "None";
            break;
          case ydf::model::proto::GenericHyperParameterSpecification::Value::
              TYPE_NOT_SET:
            return absl::InvalidArgumentError(
                absl::Substitute("Missing type for field $0", field_name));
        }
      }

      // If the parameter is conditional on a parent parameter values, and the
      // default value of the parent parameter does not satisfy the condition,
      // the default value is set to None.
      if (field_def.has_conditional()) {
        const auto& conditional = field_def.conditional();
        const auto& parent_field =
            specifications.fields().find(conditional.control_field());
        if (parent_field == specifications.fields().end()) {
          return absl::InvalidArgumentError(
              absl::Substitute("Unknown conditional field $0 for field $1",
                               conditional.control_field(), field_name));
        }
        ASSIGN_OR_RETURN(const auto condition,
                         ydf::utils::SatisfyDefaultCondition(
                             parent_field->second, conditional));
        if (!condition) {
          attr_py_default_value = "None";
        }
      }

      // Constructor argument.
      absl::SubstituteAndAppend(&fields_constructor,
                                "      $0: Optional[$1] = $2", field_name,
                                attr_py_type, attr_py_default_value);

      // Assignation to parameter dictionary.
      absl::SubstituteAndAppend(
          &fields_dict, "                      \"$0\" : $0,\n", field_name);

      // Documentation
      if (field_def.documentation().description().empty()) {
        // Refer to the proto.
        absl::SubstituteAndAppend(&fields_documentation, "    $0: See $1\n",
                                  field_name,
                                  field_def.documentation().proto_path());
      } else {
        // Actual documentation.
        absl::StrAppend(
            &fields_documentation,
            FormatDocumentation(
                absl::StrCat(field_name, ": ",
                             field_def.documentation().description(),
                             " Default: ", attr_py_default_value, "."),
                /*leading_spaces_first_line=*/4,
                /*leading_spaces_next_lines=*/6));
      }
    }

    // Pre-configured hyper-parameters.
    std::string predefined_hp_doc;
    std::string predefined_hp_list;
    ASSIGN_OR_RETURN(std::tie(predefined_hp_doc, predefined_hp_list),
                     BuildPredefinedHyperParameter(learner.get()));

    const auto free_text_documentation =
        FormatDocumentation(specifications.documentation().description(),
                            /*leading_spaces_first_line=*/2 - 2,
                            /*leading_spaces_next_lines=*/2);

    const auto nice_learner_name = LearnerKeyToNiceLearnerName(learner_key);

    absl::SubstituteAndAppend(&wrapper, R"(
class $0(core.CoreModel):
  r"""$6 learning algorithm.

  $5
  Usage example:

  ```python
  import tensorflow_decision_forests as tfdf
  import pandas as pd

  dataset = pd.read_csv("project/dataset.csv")
  tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")

  model = tfdf.keras.$0()
  model.fit(tf_dataset)

  print(model.summary())
  ```

  Attributes:
    task: Task to solve (e.g. Task.CLASSIFICATION, Task.REGRESSION,
      Task.RANKING).
    features: Specify the list and semantic of the input features of the model.
      If not specified, all the available features will be used. If specified
      and if `exclude_non_specified_features=True`, only the features in
      `features` will be used by the model. If "preprocessing" is used,
      `features` corresponds to the output of the preprocessing. In this case,
      it is recommended for the preprocessing to return a dictionary of tensors.
    exclude_non_specified_features: If true, only use the features specified in
      `features`.
    preprocessing: Functional keras model or @tf.function to apply on the input
      feature before the model to train. This preprocessing model can consume
      and return tensors, list of tensors or dictionary of tensors. If
      specified, the model only "sees" the output of the preprocessing (and not
      the raw input). Can be used to prepare the features or to stack multiple
      models on top of each other. Unlike preprocessing done in the tf.dataset,
      the operation in "preprocessing" are serialized with the model.
    ranking_group: Only for `task=Task.RANKING`. Name of a tf.string feature that
      identifies queries in a query/document ranking task. The ranking group
      is not added automatically for the set of features if
      `exclude_non_specified_features=false`.
    temp_directory: Temporary directory used during the training. The space
      required depends on the learner. In many cases, only a temporary copy of a
      model will be there.
    verbose: If true, displays information about the training.
    hyperparameter_template: Override the default value of the hyper-parameters.
      If None (default) the default parameters of the library are used. If set,
      `default_hyperparameter_template` refers to one of the following
      preconfigured hyper-parameter sets. Those sets outperforms the default
      hyper-parameters (either generally or in specific scenarios).
      You can omit the version (e.g. remove "@v5") to use the last version of
      the template. In this case, the hyper-parameter can change in between
      releases (not recommended for training in production).
    advanced_arguments: Advanced control of the model that most users won't need
      to use. See `AdvancedArguments` for details.
$7
$2

  """

  @core._list_explicit_arguments
  def __init__(self,
      task: Optional[TaskType] = core.Task.CLASSIFICATION,
      features: Optional[List[core.FeatureUsage]] = None,
      exclude_non_specified_features: Optional[bool] = False,
      preprocessing: Optional["tf.keras.models.Functional"] = None,
      ranking_group: Optional[str] = None,
      temp_directory: Optional[str] = None,
      verbose: Optional[bool] = True,
      hyperparameter_template: Optional[str] = None,
      advanced_arguments: Optional[AdvancedArguments] = None,
$3,
      explicit_args: Optional[Set[str]] = None):

    learner_params = {
$4
      }

    if hyperparameter_template is not None:
      learner_params = core._apply_hp_template(learner_params,
        hyperparameter_template, self.predefined_hyperparameters(),
        explicit_args)

    super($0, self).__init__(task=task,
      learner="$1",
      learner_params=learner_params,
      features=features,
      exclude_non_specified_features=exclude_non_specified_features,
      preprocessing=preprocessing,
      ranking_group=ranking_group,
      temp_directory=temp_directory,
      verbose=verbose,
      advanced_arguments=advanced_arguments)

  @staticmethod
  def predefined_hyperparameters() -> List[core.HyperParameterTemplate]:
    return $8
)",
                              /*$0*/ class_name, /*$1*/ learner_key,
                              /*$2*/ fields_documentation,
                              /*$3*/ fields_constructor, /*$4*/ fields_dict,
                              /*$5*/ free_text_documentation,
                              /*$6*/ nice_learner_name,
                              /*$7*/ predefined_hp_doc,
                              /*$8*/ predefined_hp_list);
  }

  return wrapper;
}

}  // namespace decision_forests
}  // namespace tensorflow
