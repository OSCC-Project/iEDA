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
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ProcessCorner.hpp"
namespace itf
{

class ItfService {
 public:
  // constructor
  ItfService() = default;
  ~ItfService() = default;

  // getter
  std::vector<ProcessCorner*> get_process_corners() const;
  ProcessCorner* get_last_process_corner();
  const ProcessCorner* get_last_process_corner() const;
  
  // take ownership
  std::vector<std::unique_ptr<ProcessCorner>> take_process_corners();
  std::unique_ptr<ProcessCorner> take_last_process_corner();

  // setter
  void add_process_corner(std::unique_ptr<ProcessCorner> corner);

  // function
  ProcessCorner* find_process_corner(const std::string&) const;

 private:
  std::vector<std::unique_ptr<ProcessCorner>> _process_corners;
};

//////// inline /////////

inline void
ItfService::add_process_corner(std::unique_ptr<ProcessCorner> c) 
{
  if (c) _process_corners.push_back(std::move(c));
}

inline ProcessCorner* 
ItfService::find_process_corner(const std::string& process_name) const
{
  for (const auto& c : _process_corners) {
    if (c && c->get_technology() == process_name) {
      return c.get();
    }
  }

  return nullptr;
}

inline ProcessCorner*
ItfService::get_last_process_corner()
{
  return _process_corners.size() ? _process_corners.back().get() : nullptr;
}

inline const ProcessCorner*
ItfService::get_last_process_corner() const
{
  return _process_corners.size() ? _process_corners.back().get() : nullptr;
}

inline std::vector<std::unique_ptr<ProcessCorner>>
ItfService::take_process_corners()
{
  auto ret = std::move(_process_corners);
  _process_corners.clear();
  return ret;
}

inline std::unique_ptr<ProcessCorner>
ItfService::take_last_process_corner()
{
  if (_process_corners.empty()) {
    return nullptr;
  }

  auto ret = std::move(_process_corners.back());
  _process_corners.pop_back();
  return ret;
}

inline std::vector<ProcessCorner*>
ItfService::get_process_corners() const
{
  std::vector<ProcessCorner*> process_corners;
  process_corners.reserve(_process_corners.size());
  for (const auto& corner : _process_corners) {
    process_corners.push_back(corner.get());
  }
  return process_corners;
}



} // namespace itf
