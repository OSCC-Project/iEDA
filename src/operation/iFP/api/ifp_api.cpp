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
#include "ifp_api.h"

#include "init_design.h"
#include "io_placer.h"
#include "tapcell.h"

namespace ifp {
FpApi* FpApi::_instance = nullptr;

/**
 * @brief
 *
 * @param die_lx
 * @param die_ly
 * @param die_ux
 * @param die_uy
 * @return true
 * @return false
 */
bool FpApi::initDie(double die_lx, double die_ly, double die_ux, double die_uy)
{
  InitDesign init_desing;

  return init_desing.initDie(die_lx, die_ly, die_ux, die_uy);
}
/**
 * @brief
 *
 * @param core_lx
 * @param core_ly
 * @param core_ux
 * @param core_uy
 * @param core_site_name
 * @param iocell_site_name
 * @return true
 * @return false
 */
bool FpApi::initCore(double core_lx, double core_ly, double core_ux, double core_uy, std::string core_site_name,
                     std::string iocell_site_name, std::string corner_site_name)
{
  InitDesign init_desing;

  return init_desing.initCore(core_lx, core_ly, core_ux, core_uy, core_site_name, iocell_site_name, corner_site_name);
}

/**
 * @brief
 *
 * @param layer_name
 * @param x_offset
 * @param x_pitch
 * @param y_offset
 * @param y_pitch
 * @return true
 * @return false
 */
bool FpApi::makeTracks(std::string layer_name, int x_offset, int x_pitch, int y_offset, int y_pitch)
{
  InitDesign init_desing;

  return init_desing.makeTracks(layer_name, x_offset, x_pitch, y_offset, y_pitch);
}
/**
 * @brief auto place io pins
 *
 * @param layer_name
 * @param width
 * @param height
 * @return true
 * @return false
 */
bool FpApi::autoPlacePins(std::string layer_name, int width, int height, std::vector<std::string> sides)
{
  IoPlacer io_placer;

  return io_placer.autoPlacePins(layer_name, width, height, sides);
}

bool FpApi::placePort(std::string pin_name, int32_t x_offset, int32_t y_offset, int32_t rect_width, int32_t rect_height,
                      std::string layer_name)
{
  IoPlacer io_placer;

  return io_placer.placePort(pin_name, x_offset, y_offset, rect_width, rect_height, layer_name);
}

bool FpApi::autoPlacePad(std::vector<std::string> pad_masters, std::vector<std::string> conner_masters)
{
  IoPlacer io_placer;

  return io_placer.autoPlacePad(pad_masters, conner_masters);
}

bool FpApi::placeIOFiller(std::vector<std::string> filler_name_list, std::string prefix)
{
  IoPlacer io_placer;

  return io_placer.autoIOFiller(filler_name_list, prefix);
}

bool FpApi::tapCells(double distance, std::string tapcell_name, std::string endcap_name)
{
  TapCellPlacer tapcell;

  return tapcell.tapCells(distance, tapcell_name, endcap_name);
}

}  // namespace ifp
