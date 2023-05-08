#pragma once

#include "ids.hpp"

namespace ito {

class Master {
 public:
  Master() = default;
  Master(idb::IdbCellMaster *idb_master);

  ~Master() = default;

  // bool isAutoPlaceable() { return _is_auto_placeable; }

  unsigned int get_width() const { return _width; }
  unsigned int get_height() const { return _height; }

 private:
  unsigned int _width;
  unsigned int _height;

  // bool _is_auto_placeable = false;
};

} // namespace ito