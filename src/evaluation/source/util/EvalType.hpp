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
#ifndef SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALTYPE_HPP_
#define SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALTYPE_HPP_

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace eval {
enum class NET_TYPE
{
  kNone = 0,
  kSignal = 1,
  kClock = 2,
  kReset = 3,
  kFakeNet = 4
};

enum class WIRELENGTH_TYPE
{
  kNone = 0,
  kHPWL = 1,
  kFLUTE = 2,
  kB2B = 3,
  kEGR = 4
};

enum class PIN_TYPE
{
  kNone = 0,
  kInstancePort = 1,
  kIOPort = 2,
  kFakePin = 3
};

enum class PIN_IO_TYPE
{
  kNone = 0,
  kInput = 1,
  kOutput = 2,
  kInputOutput = 3
};

enum class INSTANCE_LOC_TYPE
{
  kNone = 0,
  kNormal = 1,
  kOutside = 2,
  kFakeInstance = 3
};

enum class INSTANCE_STATUS
{
  kNone = 0,
  kFixed = 1,
  kCover = 2,
  kPlaced = 3,
  kUnplaced = 4,
  kMax
};

enum class CONGESTION_TYPE
{
  kNone = 0,
  kInstDens = 1,
  kPinDens = 2,
  kNetCong = 3,
  kGRCong = 4
};

enum class CHIP_REGION_TYPE
{
  kNone = 0,
  kDie = 1,
  kCore = 2
};

enum class RUDY_TYPE
{
  kNone = 0,
  kRUDY = 1,
  kPinRUDY = 2,
  kLUTRUDY = 3
};

enum class DIRECTION
{
  kNone = 0,
  kH = 1,
  kV = 2
};

enum class NET_CONNECT_TYPE
{
  kNone = 0,
  kSignal = 1,
  kClock = 2,
  kPower = 3,
  kGround = 4
};

class EvalPropertyMap
{
 public:
  EvalPropertyMap()
  {
    _rudy_type_list
        = {{RUDY_TYPE::kNone, ""}, {RUDY_TYPE::kRUDY, "rudy"}, {RUDY_TYPE::kPinRUDY, "pin_rudy"}, {RUDY_TYPE::kLUTRUDY, "lut_rudy"}};

    _instance_status_list = {{INSTANCE_STATUS::kNone, ""},
                             {INSTANCE_STATUS::kFixed, "fixed"},
                             {INSTANCE_STATUS::kCover, "cover"},
                             {INSTANCE_STATUS::kPlaced, "placed"},
                             {INSTANCE_STATUS::kUnplaced, "unplaced"}};

    _congestion_type_list = {{CONGESTION_TYPE::kNone, ""},
                             {CONGESTION_TYPE::kInstDens, "instance_density"},
                             {CONGESTION_TYPE::kPinDens, "pin_density"},
                             {CONGESTION_TYPE::kNetCong, "net_congestion"},
                             {CONGESTION_TYPE::kGRCong, "gr_congestion"}};
  }
  ~EvalPropertyMap() = default;

  INSTANCE_STATUS get_instance_status(std::string status_name)
  {
    auto result = std::find_if(_instance_status_list.begin(), _instance_status_list.end(),
                               [status_name](const auto& iter) { return iter.second == status_name; });

    if (result == _instance_status_list.end()) {
      return INSTANCE_STATUS::kNone;
    }

    return result->first;
  }
  std::string get_instance_status_str(INSTANCE_STATUS status)
  {
    auto iter = _instance_status_list.find(status);
    if (iter == _instance_status_list.end()) {
      return std::string("");
    }

    return iter->second;
  }

  CONGESTION_TYPE get_congestion_type(std::string congestion_type_name)
  {
    auto result = std::find_if(_congestion_type_list.begin(), _congestion_type_list.end(),
                               [congestion_type_name](const auto& iter) { return iter.second == congestion_type_name; });

    if (result == _congestion_type_list.end()) {
      return CONGESTION_TYPE::kNone;
    }

    return result->first;
  }
  std::string get_congestion_type_str(CONGESTION_TYPE congestion_type)
  {
    auto iter = _congestion_type_list.find(congestion_type);
    if (iter == _congestion_type_list.end()) {
      return std::string("");
    }

    return iter->second;
  }

  RUDY_TYPE get_rudy_type(std::string rudy_type_name)
  {
    auto result = std::find_if(_rudy_type_list.begin(), _rudy_type_list.end(),
                               [rudy_type_name](const auto& iter) { return iter.second == rudy_type_name; });

    if (result == _rudy_type_list.end()) {
      return RUDY_TYPE::kNone;
    }

    return result->first;
  }
  std::string get_rudy_type_str(RUDY_TYPE rudy_type)
  {
    auto iter = _rudy_type_list.find(rudy_type);
    if (iter == _rudy_type_list.end()) {
      return std::string("");
    }

    return iter->second;
  }

 private:
  std::map<RUDY_TYPE, std::string> _rudy_type_list;
  std::map<INSTANCE_STATUS, std::string> _instance_status_list;
  std::map<CONGESTION_TYPE, std::string> _congestion_type_list;
};

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALTYPE_HPP_
