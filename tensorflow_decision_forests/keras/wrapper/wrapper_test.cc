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

#include "gtest/gtest.h"

namespace tensorflow {
namespace decision_forests {
namespace {

TEST(KerasLearnerWrappers, LearnerKeyToClassName) {
  EXPECT_EQ(LearnerKeyToClassName("RANDOM_FOREST"), "RandomForestModel");
}

TEST(KerasLearnerWrappers, Base) {
  std::cout << GenKerasPythonWrapper().value() << std::endl;
}

TEST(KerasLearnerWrappers, FormatDocumentation) {
  const auto formatted = FormatDocumentation(R"(AAA AAA AAA AAA AAA.
AAA AAA AAA AAA.
- AAA AAA AAA AAA.
- AAA AAA AAA AAA.
AAA AAA AAA AAA.
  AAA AAA AAA AAA.)",
                                             /*leading_spaces_first_line=*/4,
                                             /*leading_spaces_next_lines=*/6);
  EXPECT_EQ(formatted, R"(    AAA AAA AAA AAA AAA.
      AAA AAA AAA AAA.
      - AAA AAA AAA AAA.
      - AAA AAA AAA AAA.
      AAA AAA AAA AAA.
          AAA AAA AAA AAA.
)");
}

TEST(KerasLearnerWrappers, NumLeadingSpaces) {
  EXPECT_EQ(NumLeadingSpaces(""), 0);
  EXPECT_EQ(NumLeadingSpaces(" "), 1);
  EXPECT_EQ(NumLeadingSpaces("  "), 2);
  EXPECT_EQ(NumLeadingSpaces("  HELLO "), 2);
}

}  // namespace
}  // namespace decision_forests
}  // namespace tensorflow
