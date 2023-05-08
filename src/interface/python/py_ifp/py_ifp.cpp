#include "py_ifp.h"

#include <idm.h>
#include <ifp_api.h>
#include <tool_manager.h>

#include <Str.hh>

namespace python_interface {

bool fpInit(const std::string& die_area, const std::string& core_area, const std::string& core_site, const std::string& io_site)
{
  auto die = ieda::Str::splitDouble(die_area.c_str(), " ");
  auto core = ieda::Str::splitDouble(core_area.c_str(), " ");
  fpApiInst->initDie(die[0], die[1], die[2], die[3]);
  fpApiInst->initCore(core[0], core[1], core[2], core[3], core_site, io_site);
  return true;
}

bool fpMakeTracks(const std::string& layer, int x_start, int x_step, int y_start, int y_step)
{
  bool make_ok = fpApiInst->makeTracks(layer, x_start, x_step, y_start, y_step);
  return make_ok;
}

bool fpPlacePins(const std::string& layer, int width, int height)
{
  bool place_ok = fpApiInst->autoPlacePins(layer, width, height);
  return place_ok;
}

bool fpPlacePort(const std::string& pin_name, int offset_x, int offset_y, int width, int height, const std::string& layer)
{
  bool place_ok = fpApiInst->placePort(pin_name, offset_x, offset_y, width, height, layer);
  return place_ok;
}

bool fpPlaceIOFiller(std::vector<std::string>& filler_types, const std::string& prefix, const std::string& orient, double begin, double end,
                     const std::string& source)
{
  bool place_ok = fpApiInst->placeIOFiller(filler_types, prefix, orient, begin, end, source);
  return place_ok;
}

bool fpAddPlacementBlockage(const std::string& box)
{
  auto blk = ieda::Str::splitInt(box.c_str(), " ");
  int32_t llx = blk[0];
  int32_t lly = blk[1];
  int32_t urx = blk[2];
  int32_t ury = blk[3];

  dmInst->addPlacementBlockage(llx, lly, urx, ury);
  return true;
}

bool fpAddPlacementHalo(const std::string& inst_name, const std::string& distance)
{
  auto distance_val = ieda::Str::splitInt(distance.c_str(), " ");
  int32_t left = distance_val[0];
  int32_t bottom = distance_val[1];
  int32_t right = distance_val[2];
  int32_t top = distance_val[3];

  dmInst->addPlacementHalo(inst_name, top, bottom, left, right);
  return true;
}

bool fpAddRoutingBlockage(const std::string& layer, const std::string& box, bool exceptpgnet)
{
  auto layers = Str::split(layer.c_str(), " ");
  auto box_result = Str::splitInt(box.c_str(), " ");

  int32_t llx = box_result[0];
  int32_t lly = box_result[1];
  int32_t urx = box_result[2];
  int32_t ury = box_result[3];

  dmInst->addRoutingBlockage(llx, lly, urx, ury, layers, exceptpgnet);
  return true;
}

bool fpAddRoutingHalo(const std::string& layer, const std::string& distance, bool exceptpgnet, const std::string& inst_name)
{
  auto layers = Str::split(layer.c_str(), " ");
  auto box_result = Str::splitInt(distance.c_str(), " ");
  int32_t left = box_result[0];
  int32_t bottom = box_result[1];
  int32_t right = box_result[2];
  int32_t top = box_result[3];

  dmInst->addRoutingHalo(inst_name, layers, top, bottom, left, right, exceptpgnet);
  return true;
}

bool fpTapCell(const std::string& tapcell, double distance, const std::string& endcap)
{
  fpApiInst->insertTapCells(distance, tapcell);
  fpApiInst->insertEndCaps(endcap);
  return true;
}
}  // namespace python_interface