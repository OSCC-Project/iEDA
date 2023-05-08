#pragma once

#include <string>
#include <vector>

namespace idb {
class IdbLayer;
class IdbSpecialWireSegment;

class IdbRect;
class IdbVia;
class IdbLayerCut;

class IdbSpecialNet;
class IdbLayerRouting;
class IdbSpecialWire;

enum class SegmentType : int8_t;
enum class IdbWireShapeType : uint8_t;
enum class IdbOrient : uint8_t;

template <typename T>
class IdbCoordinate;
}  // namespace idb

namespace ipdn {
class PdnVia
{
 public:
  explicit PdnVia() {}
  ~PdnVia() {}

  /// operator

  idb::IdbVia* findVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design);
  idb::IdbVia* createVia(idb::IdbLayerCut* layer_cut, int32_t width_design, int32_t height_design, std::string via_name);
  idb::IdbSpecialWireSegment* createSpecialWireVia(idb::IdbLayer* layer, int32_t route_width, idb::IdbWireShapeType wire_shape_type,
                                                   idb::IdbCoordinate<int32_t>* coord, idb::IdbVia* via);

  bool addSegmentVia(std::string net_name, std::string cut_layer_name, int32_t coord_x, int32_t coord_y, int32_t width, int32_t height);
  bool addSegmentVia(std::string net_name, std::string top_metal, std::string bottom_metal, int32_t coord_x, int32_t coord_y, int32_t width,
                     int32_t height);

  bool addSegmentVia(std::string net_name, std::string top_metal, std::string bottom_metal, double coord_x, double coord_y, int32_t width,
                     int32_t height);

 private:
  int32_t transUnitDB(double value);
};

}  // namespace ipdn