#pragma once

#include "NoConfig.h"
#include "json.hpp"
#include <mutex>

namespace ino {
using Json = nlohmann::json;

class JsonParser {
 public:
  static JsonParser *get_json_parser();

  void parse(const string &json_file, NoConfig *config) const;

 private:
  JsonParser() = default;
  JsonParser(const JsonParser &parser) = delete;
  JsonParser &operator=(const JsonParser &) = default;

  void jsonToConfig(Json *json, NoConfig *config) const;

  void printConfig(NoConfig *config) const;

  static JsonParser *_json_parser;
};
} // namespace ino