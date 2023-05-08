#pragma once

#include <string>
#include <vector>

namespace python_interface {

bool fpInit(const std::string& die_area, const std::string& core_area, const std::string& core_site, const std::string& io_site);
bool fpMakeTracks(const std::string& layer, int x_start, int x_step, int y_start, int y_step);
bool fpPlacePins(const std::string& layer, int width, int height);
bool fpPlacePort(const std::string& pin_name, int offset_x, int offset_y, int width, int height, const std::string& layer);
bool fpPlaceIOFiller(std::vector<std::string>& filler_types, const std::string& prefix, const std::string& orient, double begin, double end,
                     const std::string& source);
bool fpAddPlacementBlockage(const std::string& box);
bool fpAddPlacementHalo(const std::string& inst_name, const std::string& distance);
bool fpAddRoutingBlockage(const std::string& layer, const std::string& box, bool exceptpgnet);
bool fpAddRoutingHalo(const std::string& layer, const std::string& distance, bool exceptpgnet, const std::string& inst_name);
bool fpTapCell(const std::string& tapcell, double distance, const std::string& endcap);
}  // namespace python_interface