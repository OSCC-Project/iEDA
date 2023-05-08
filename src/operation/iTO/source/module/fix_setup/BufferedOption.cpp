#include "BufferedOption.h"
#include "api/TimingEngine.hh"

namespace ito {
Required BufferedOption::get_required_arrival_time() { return _req - _required_delay; }

void BufferedOption::printBuffered(int level) {
  printTree(level);
  switch (_type) {
  case BufferedOptionType::kLoad: {
    break;
  }
  case BufferedOptionType::kBuffer:
  case BufferedOptionType::kWire: {
    _left->printBuffered(level + 1);
    break;
  }
  case BufferedOptionType::kJunction: {
    if (_left) {
      _left->printBuffered(level + 1);
    }
    if (_right) {
      _right->printBuffered(level + 1);
    }
    break;
  }
  }
}

void BufferedOption::printTree(int level) {
  switch (_type) {
  case BufferedOptionType::kLoad: {
    printf("%*s load %s (%d, %d) cap %f req %lf\n", level, "", _load_pin->get_name(),
           _location.get_x(), _location.get_y(), _cap, _req);
    break;
  }
  case BufferedOptionType::kBuffer: {
    printf("%*s buffer (%d, %d) %s cap %f req %lf\n", level, "", _location.get_x(),
           _location.get_y(), _buffer_cell->get_cell_name(), _cap,
           get_required_arrival_time());
    break;
  }
  case BufferedOptionType::kWire: {
    printf("%*s wire (%d, %d) cap %f req %lf\n", level, "", _location.get_x(),
           _location.get_y(), _cap, get_required_arrival_time());
    break;
  }
  case BufferedOptionType::kJunction: {
    printf("%*s junction (%d, %d) cap %f req %lf\n", level, "", _location.get_x(),
           _location.get_y(), _cap, get_required_arrival_time());
    break;
  }
  }
}

} // namespace ito