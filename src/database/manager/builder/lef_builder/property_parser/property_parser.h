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

#include <math.h>

#include <iostream>
#include <string>

#include "lef_service.h"

namespace idb {

enum class LefPropertyType
{
  kNone,
  kCutSpacing,
  kMax
};
template <typename T>
class PropertyBaseParser
{
 public:
  explicit PropertyBaseParser(IdbLefService* lef_service) { _lef_service = lef_service; }
  virtual ~PropertyBaseParser() { _lef_service = nullptr; }

  /// operator
  IdbLefService* get_lef_service() { return _lef_service; }
  IdbLayout* get_layout() { return _lef_service != nullptr ? _lef_service->get_layout() : nullptr; }

  int32_t transAreaDB(double value) { return _lef_service->get_layout()->transAreaDB(value); }
  int32_t transUnitDB(double value) { return _lef_service->get_layout()->transUnitDB(value); }

  virtual bool parse(const std::string& name, const std::string& value, T* data) = 0;

 private:
  IdbLefService* _lef_service;
};

}  // namespace idb
