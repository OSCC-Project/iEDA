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
#include "property_parser.h"

namespace idb {
class CutLayerParser : public PropertyBaseParser<IdbLayerCut>
{
 public:
  explicit CutLayerParser(IdbLefService* lef_service) : PropertyBaseParser(lef_service) {}
  ~CutLayerParser() override = default;

  /// operator
  bool parse(const std::string& name, const std::string& value, IdbLayerCut* data) override;

 private:
  bool parse_lef58_cutclass(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_enclosure(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_enclosureedge(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_eolenclosure(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_eolspacing(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_spacingtable(const std::string& value, IdbLayerCut* data);
};

}  // namespace idb
