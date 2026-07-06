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
#include <fstream>
#include <sstream>

#include "MappingBuilder.hpp"
#include "log/Log.hh"
namespace ircx {
namespace parser {

void MappingBuilder::clear()
{
  design_to_process_layer_names_.clear();
  process_to_design_layer_names_.clear();
}

bool MappingBuilder::read(const std::string& mappingPath)
{
  clear();

  std::ifstream mappingFile(mappingPath);
  if (!mappingFile.is_open()) {
    LOG_ERROR << "Failed to open RCX mapping file: " << mappingPath;
    return false;
  }

  std::string line;
  while (std::getline(mappingFile, line)) {
    std::string designLayerName;
    std::string processLayerName;
    if (std::istringstream(line) >> designLayerName >> processLayerName) {
      design_to_process_layer_names_[designLayerName] = processLayerName;
      process_to_design_layer_names_[processLayerName] = designLayerName;
    }
  }

  return true;
}

}  // namespace parser
}  // namespace ircx
