// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file ConfigTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-06-01
 * @brief CTS runtime configuration parser contract tests.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "database/config/Config.hh"

namespace icts_test {
namespace {

auto writeConfigFile(const std::string& file_name, const std::string& content) -> std::filesystem::path
{
  const auto config_dir = std::filesystem::temp_directory_path() / "icts_config_contract_test";
  std::filesystem::create_directories(config_dir);
  const auto config_path = config_dir / file_name;
  std::ofstream output_stream(config_path);
  output_stream << content;
  return config_path;
}

auto containsText(const std::vector<std::string>& values, const std::string& expected_text) -> bool
{
  for (const auto& value : values) {
    if (value.find(expected_text) != std::string::npos) {
      return true;
    }
  }
  return false;
}

TEST(ConfigTest, MissingOptionalValuesUseDefaults)
{
  const auto config_path = writeConfigFile("defaults.json", "{}");
  icts::Config config;

  EXPECT_TRUE(config.init(config_path.string()));
  EXPECT_EQ(config.get_max_fanout(), 32U);
  EXPECT_DOUBLE_EQ(config.get_skew_bound(), 0.04);
  EXPECT_TRUE(config.get_warnings().empty());
  EXPECT_TRUE(config.get_last_error().empty());
}

TEST(ConfigTest, UnknownAndDeprecatedKeysWarnAndDoNotFail)
{
  const auto config_path = writeConfigFile("deprecated_and_unknown.json", R"json({
    "skew_bound": "0.10",
    "use_netlist": "OFF",
    "net_list": [],
    "unexpected_key": 7
  })json");
  icts::Config config;

  EXPECT_TRUE(config.init(config_path.string()));
  EXPECT_DOUBLE_EQ(config.get_skew_bound(), 0.10);
  EXPECT_TRUE(config.get_last_error().empty());
  EXPECT_EQ(config.get_warnings().size(), 3U);
  EXPECT_TRUE(containsText(config.get_warnings(), "config item \"use_netlist\": this item is no longer used"));
  EXPECT_TRUE(containsText(config.get_warnings(), "config item \"net_list\": this item is no longer used"));
  EXPECT_TRUE(containsText(config.get_warnings(), "invalid config key \"unexpected_key\""));
}

TEST(ConfigTest, InvalidPresentNumericValueFails)
{
  const auto config_path = writeConfigFile("invalid_numeric.json", R"json({
    "max_fanout": "not_an_integer"
  })json");
  icts::Config config;

  EXPECT_FALSE(config.init(config_path.string()));
  EXPECT_NE(config.get_last_error().find("invalid unsigned integer value for key \"max_fanout\""), std::string::npos);
}

TEST(ConfigTest, InvalidRoutingLayerItemFails)
{
  const auto config_path = writeConfigFile("invalid_routing_layer.json", R"json({
    "routing_layer": [5, "bad_layer"]
  })json");
  icts::Config config;

  EXPECT_FALSE(config.init(config_path.string()));
  EXPECT_NE(config.get_last_error().find("invalid unsigned integer list value for key \"routing_layer\""), std::string::npos);
}

}  // namespace
}  // namespace icts_test
