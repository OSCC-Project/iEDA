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

#include <string>
#include <vector>

namespace python_interface {

bool fpInit(const std::string& die_area, const std::string& core_area, const std::string& core_site, const std::string& io_site,
            const std::string& corner_site);
bool fpMakeTracks(const std::string& layer, int x_start, int x_step, int y_start, int y_step);
bool fpPlacePins(const std::string& layer, int width, int height, std::vector<std::string>& sides);
bool fpPlacePort(const std::string& pin_name, int offset_x, int offset_y, int width, int height, const std::string& layer);
bool fpPlaceIOFiller(std::vector<std::string>& filler_types, const std::string& prefix);
bool fpAddPlacementBlockage(const std::string& box);
bool fpAddPlacementHalo(const std::string& inst_name, const std::string& distance);
bool fpAddRoutingBlockage(const std::string& layer, const std::string& box, bool exceptpgnet);
bool fpAddRoutingHalo(const std::string& layer, const std::string& distance, bool exceptpgnet, const std::string& inst_name);
bool fpTapCell(const std::string& tapcell, double distance, const std::string& endcap);
}  // namespace python_interface