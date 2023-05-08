#ifndef IDRC_SRC_DB_IDRWRAPPER_H_
#define IDRC_SRC_DB_IDRWRAPPER_H_

#include "DRCCOMUtil.h"
// #include "Database.h"
#include "DrcConfig.h"
#include "DrcDesign.h"
// #include "RRNet.h"
// #include "RRSubNet.h"
#include "RegionQuery.h"

namespace idrc {
class IDRWrapper
{
 public:
  IDRWrapper(DrcConfig* config, Tech* tech, DrcDesign* drc_design, RegionQuery* region_query)
      : _config(config), _tech(tech), _drc_design(drc_design), _region_query(region_query)
  {
  }
  IDRWrapper(const IDRWrapper& other) = delete;
  IDRWrapper(IDRWrapper&& other) = delete;
  ~IDRWrapper() {}
  IDRWrapper& operator=(const IDRWrapper& other) = delete;
  IDRWrapper& operator=(IDRWrapper&& other) = delete;

  // setter
  // void set_dr_design(idr::Database* dr_design) { _dr_design = dr_design; }
  // // getter
  // idr::Database* get_dr_design() { return _dr_design; }
  // Tech* get_tech() { return _tech; }
  DrcDesign* get_drc_design() { return _drc_design; }
  DrcConfig* get_config() { return _config; }
  // // function
  // void inputAndInitFixedShapes(idr::Database* dr_design);
  // // void inputAndInitRoutingShapes(std::vector<idr::RRSubNet>& sub_net_list);
  // // RRSubNet
  // void inputAndInitCurrentRoutingShapes(std::vector<idr::RRSubNet>& sub_net_list);
  // void inputAndInitBestRoutingShapes(std::vector<idr::RRSubNet>& sub_net_list);
  // // RRNet
  // void inputAndInitRoutingShapesInRRNetList(std::vector<idr::RRNet>& rr_net_list);

  DrcNet* get_drc_net(int netId);

 protected:
  // init for region
  /// init fix shapes
  void wrapPinListAndBlockageList();
  void wrapBlockageList();
  void wrapNetPinList();
  // init routing shapes
  // void wrapSegmentsAndVias(DrcNet* drc_net, idr::RRSubNet& sub_net);
  // void wrapSegmentsAndVias(DrcNet* drc_net, std::vector<idr::SpaceSegment<idr::RRPoint>>& dr_space_segment_list);
  // init for all layout
  // void wrapBlockageList();
  // void wrapNetList();
  // void wrapRoutingLayerList();
  // void wrapViaLib();

 private:
  DrcConfig* _config = nullptr;
  Tech* _tech = nullptr;
  DrcDesign* _drc_design = nullptr;
  // idr::Database* _dr_design = nullptr;
  RegionQuery* _region_query = nullptr;
  std::map<int, DrcNet*> _id_to_net;

  // function
  // void wrapBlockage(idr::Blockage& dr_block, DrcRect* drc_block);
  // void wrapNetPinList(idr::Net& dr_net, DrcNet* drc_net);

  // void wrapRect(DrcRectangle<int>& rect, idr::Rectangle<int>& dr_rect);
  // void wrapRect(DrcRect* drc_rect, idr::Rectangle<int>& dr_rect);

  // void wrapNetVia(idr::SpaceSegment<idr::RRPoint>& dr_space_segment, DrcNet* drc_net);
  // DrcRect* getViaMetalRect(int center_x, int center_y, DrcEnclosure& enclosure);
  // void wrapNetSegment(idr::SpaceSegment<idr::RRPoint>& dr_space_segment, DrcNet* drc_net);
};
}  // namespace idrc

#endif