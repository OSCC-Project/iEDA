#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "CtsConfig.h"

namespace icts {

using std::string;
using std::vector;

class JsonParser {
 public:
  static JsonParser &getInstance();

  void parse(const string &json_file, CtsConfig *config) const;

 private:
  JsonParser() = default;
  JsonParser(const JsonParser &parser) = delete;
  JsonParser &operator=(const JsonParser &) = default;
};
}  // namespace icts