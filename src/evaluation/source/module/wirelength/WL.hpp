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
#ifndef SRC_EVALUATION_SOURCE_MODULE_WIRELENGTH_WL_HPP_
#define SRC_EVALUATION_SOURCE_MODULE_WIRELENGTH_WL_HPP_

#include "WLNet.hpp"

namespace eval {
class WL
{
 public:
  virtual ~WL() {}
  virtual int64_t getTotalWL(const std::vector<WLNet*>& net_list) = 0;
};

class WLMWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class HPWLWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class HTreeWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class VTreeWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class CliqueWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class StarWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class B2BWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class FluteWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class PlaneRouteWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class SpaceRouteWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};

class DRWL : public WL
{
 public:
  int64_t getTotalWL(const std::vector<WLNet*>& net_list);
};
}  // namespace eval

#endif // SRC_EVALUATION_SOURCE_MODULE_WIRELENGTH_WL_HPP_
