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
#include "property_parser.h"

namespace idb {
class RoutingLayerParser : public PropertyBaseParser<IdbLayerRouting>
{
 public:
  explicit RoutingLayerParser(IdbLefService* lefservice) : PropertyBaseParser(lefservice) {}
  ~RoutingLayerParser() override = default;

  // operator
  bool parse(const std::string& name, const std::string& value, IdbLayerRouting* data) override;

 private:
  bool parse_lef58_area(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_conerfillspacing(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_minimuncut(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_minstep(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_spacing(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_spacingtable(const std::string& value, IdbLayerRouting* data);

  bool parse_lef58_spacingtable_jogtojog(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_spacing_eol(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_spacing_notchlength(const std::string& value, IdbLayerRouting* data);
};
}  // namespace idb
