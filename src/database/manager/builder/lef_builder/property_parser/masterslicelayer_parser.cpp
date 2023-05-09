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
#include "masterslicelayer_parser.h"
#include "lef58_property/masterslicelayer_property_parser.h"

namespace idb {

bool MastersliceLayerParser::parse(const std::string& name, const std::string& value, IdbLayerMasterslice* data)
{
    if(name == "LEF58_TYPE"){
        return parse_lef58_type(value, data);
    }
    std::cout << "Unhandled PROPERTY: " << name << " \"" << value << "\"" << std::endl;
    return false;

}

bool MastersliceLayerParser::parse_lef58_type(const std::string& value, IdbLayerMasterslice* data)
{
    std::string type;
    bool ok = masterslicelayer_property::parse_lef58_type(value.begin(), value.end(), type);
    if(not ok){
        return false;
    }
    data->set_lef58_type(std::move(type));
    return true;

}
}  // namespace idb