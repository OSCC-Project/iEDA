#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>

#include "json.hpp"

int32_t main()
{
  std::ofstream json_file("test.json");
  ///////////////////////////
  ///////////////////////////
  ///////////////////////////

  nlohmann::json top_json;
  for (size_t i = 0; i < 10; i++) {
    nlohmann::json net_json;
    net_json["net_name"] = "CK";
    net_json["result"].push_back("aaa");
    net_json["result"].push_back("bbb");
    net_json["patch"].push_back("aaa");
    net_json["patch"].push_back("bbb");
    top_json.push_back(net_json);
  }

  ///////////////////////////
  ///////////////////////////
  ///////////////////////////
  json_file << top_json;
  json_file.close();
  return 0;
}