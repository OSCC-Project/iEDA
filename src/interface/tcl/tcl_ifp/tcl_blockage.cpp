#include "Str.hh"
#include "idm.h"
#include "ifp_api.h"
#include "tcl_ifp.h"
#include "tool_manager.h"

namespace tcl {

TclFpAddPlacementBlockage::TclFpAddPlacementBlockage(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* box = new TclStringOption("-box", 0, nullptr);
  addOption(box);
}

unsigned TclFpAddPlacementBlockage::check()
{
  // TclOption* box = getOptionOrArg("-box");
  return 1;
}

unsigned TclFpAddPlacementBlockage::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* box = getOptionOrArg("-box");

  auto box_shape = box->getStringVal();
  Str str = Str();
  std::vector<int32_t> blk = str.splitInt(box_shape, " ");

  int32_t llx = blk[0];
  int32_t lly = blk[1];
  int32_t urx = blk[2];
  int32_t ury = blk[3];

  dmInst->addPlacementBlockage(llx, lly, urx, ury);

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclFpAddPlacementHalo::TclFpAddPlacementHalo(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* inst_name = new TclStringOption("-inst_name", 0, nullptr);
  auto* distance = new TclStringOption("-distance", 0, nullptr);
  addOption(distance);
  addOption(inst_name);
}

unsigned TclFpAddPlacementHalo::check()
{
  // TclOption* inst_name = getOptionOrArg("-inst_name");
  // TclOption* distance = getOptionOrArg("-distance");
  return 1;
}

unsigned TclFpAddPlacementHalo::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* inst_name = getOptionOrArg("-inst_name");
  TclOption* distance = getOptionOrArg("-distance");

  auto inst_name_val = inst_name->getStringVal();

  Str str = Str();
  std::vector<int32_t> distance_val = str.splitInt(distance->getStringVal(), " ");

  int32_t left = distance_val[0];
  int32_t bottom = distance_val[1];
  int32_t right = distance_val[2];
  int32_t top = distance_val[3];

  dmInst->addPlacementHalo(inst_name_val, top, bottom, left, right);

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TclFpAddRoutingBlockage::TclFpAddRoutingBlockage(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* layer = new TclStringOption("-layer", 0);
  auto* box = new TclStringOption("-box", 0);
  auto* pgnet = new TclIntOption("-exceptpgnet", 0);

  addOption(layer);
  addOption(box);
  addOption(pgnet);
}

unsigned TclFpAddRoutingBlockage::check()
{
  TclOption* layer = getOptionOrArg("-layer");
  TclOption* box = getOptionOrArg("-box");
  TclOption* pgnet = getOptionOrArg("-exceptpgnet");

  LOG_FATAL_IF(!layer);
  LOG_FATAL_IF(!box);
  LOG_FATAL_IF(!pgnet);

  return 1;
}

unsigned TclFpAddRoutingBlockage::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* layer = getOptionOrArg("-layer");
  TclOption* box = getOptionOrArg("-box");
  TclOption* pg = getOptionOrArg("-exceptpgnet");

  Str str = Str();

  std::vector<std::string> layer_result = str.split(layer->getStringVal(), " ");
  std::vector<int32_t> box_result = str.splitInt(box->getStringVal(), " ");
  bool exceptpgnet = (pg->getIntVal() == 1) ? true : false;

  int32_t llx = box_result[0];
  int32_t lly = box_result[1];
  int32_t urx = box_result[2];
  int32_t ury = box_result[3];

  dmInst->addRoutingBlockage(llx, lly, urx, ury, layer_result, exceptpgnet);

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclFpAddRoutingHalo::TclFpAddRoutingHalo(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* layer = new TclStringOption("-layer", 0);
  auto* distance = new TclStringOption("-distance", 0);
  auto* pgnet = new TclIntOption("-exceptpgnet", 0);
  auto* inst_name = new TclStringOption("-inst_name", 0);

  addOption(layer);
  addOption(distance);
  addOption(pgnet);
  addOption(inst_name);
}

unsigned TclFpAddRoutingHalo::check()
{
  TclOption* layer = getOptionOrArg("-layer");
  TclOption* distance = getOptionOrArg("-distance");
  // TclOption *pgnet = getOptionOrArg("-exceptpgnet");
  TclOption* inst_name = getOptionOrArg("-inst_name");

  LOG_FATAL_IF(!layer);
  LOG_FATAL_IF(!distance);
  // LOG_FATAL_IF(!pgnet);
  LOG_FATAL_IF(!inst_name);

  return 1;
}

unsigned TclFpAddRoutingHalo::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* layer = getOptionOrArg("-layer");
  TclOption* distance = getOptionOrArg("-distance");
  TclOption* pg = getOptionOrArg("-exceptpgnet");
  TclOption* inst_name = getOptionOrArg("-inst_name");

  Str str = Str();

  std::string inst_name_val = inst_name->getStringVal();

  std::vector<std::string> layer_result = str.split(layer->getStringVal(), " ");
  std::vector<int32_t> box_result = str.splitInt(distance->getStringVal(), " ");
  int32_t left = box_result[0];
  int32_t bottom = box_result[1];
  int32_t right = box_result[2];
  int32_t top = box_result[3];

  bool exceptpgnet = (pg != nullptr && pg->getIntVal() == 1) ? true : false;

  dmInst->addRoutingHalo(inst_name_val, layer_result, top, bottom, left, right, exceptpgnet);

  return 1;
}

}  // namespace tcl
