#pragma once

#include <string>
#include <vector>

#include "IdbCellMaster.h"
#include "IdbDesign.h"
#include "IdbEnum.h"
#include "ifp_enum.h"
#include "ifp_interval.h"

namespace ifp {

class IoPlacer
{
 public:
  explicit IoPlacer() {}
  ~IoPlacer() {}

  /// operator
  bool autoPlacePins(std::string layer_name, int width, int height);
  bool placePort(std::string pin_name, int32_t x_offset, int32_t y_offset, int32_t rect_width, int32_t rect_height, std::string layer_name);

  bool placeIOFiller(std::vector<std::string> filler_name_list, std::string prefix, std::string orient, double begin, double end,
                     std::string source);

  void placeIOFiller(std::vector<std::string> filler_names, const std::string prefix, Edge edge, double begin_pos, double end_pos,
                     std::string source);
  void fillInterval(Interval interval, std::vector<idb::IdbCellMaster*> fillers, const std::string prefix, std::string source);

 private:
  idb::IdbDesign* _idb_design;
  int32_t _iofiller_idx = -1;

  int32_t transUnitDB(double value);
  idb::IdbOrient transferEdgeToOrient(Edge edge);
  int32_t chooseFillerIndex(int32_t length, std::vector<idb::IdbCellMaster*> fillers);
  bool edgeIsSameToOrient(Edge edge, idb::IdbOrient orient);
  std::string transferOrientToString(idb::IdbOrient orient);

  ///

 public:
  /*****************************************************************/
  //   void initFloorplan(double die_lx, double die_ly, double die_ux, double die_uy, double core_lx, double core_ly, double core_ux,
  //                      double core_uy, string core_site_name, string iocell_site_name);

  //   void autoPlacePins(string pin_layer_name, FloorplanDb* db);
  //   void make_tracks();
  //   void make_layer_tracks(int layer, int x_offset, int x_pitch, int y_offset, int y_pitch);
  //   void set_floorplandb(FloorplanDb* floorplandb) { _floorplandb = floorplandb; }

 protected:
  //   int32_t designArea();
  //   int32_t metersToMfgGrid(double dist);

  //   void make_layer_tracks(IdbLayerRouting* routing_layer);

  //   void make_core_rows(IdbSite* site, int core_lx, int core_ly, int core_ux, int core_uy);
  //   IdbRow* createRow(string name, IdbSite* site, int32_t origin_x, int32_t origin_y, IdbOrient site_orient, int32_t num, int32_t step,
  //                     bool is_horizontal);
  //   void updateVoltageDomain(IdbSite* site, int core_lx, int core_ly, int core_ux, int core_uy);
  //   void autoPlacePins(IdbLayer* pin_layer, IdbRect& core);
};
}  // namespace ifp