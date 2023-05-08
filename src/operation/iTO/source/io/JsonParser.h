#pragma once

#include "../config/ToConfig.h"
#include <fstream>
#include <mutex>

namespace ito {
using Json = nlohmann::json;

class JsonParser {
 public:
  static JsonParser *get_json_parser();

  void parse(const string &json_file, ToConfig *config) const;

 private:
  JsonParser() = default;
  JsonParser(const JsonParser &parser) = delete;
  JsonParser &operator=(const JsonParser &) = default;

  void jsonToConfig(Json *json, ToConfig *config) const;

  void printConfig(ToConfig *config) const;

  static JsonParser *_json_parser;
};
} // namespace ito