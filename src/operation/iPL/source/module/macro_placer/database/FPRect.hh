
#pragma once

#include "FPInst.hh"

namespace ipl::imp {

class FPRect
{
 public:
  FPRect(){};
  ~FPRect(){};

  // getter
  int32_t get_x() { return _coordinate->_x; }
  int32_t get_y() { return _coordinate->_y; }
  uint32_t get_width() { return _width; }
  uint32_t get_height() { return _height; }

  // setter
  void set_x(int32_t x) { _coordinate->_x = x; }
  void set_y(int32_t y) { _coordinate->_y = y; }
  void set_width(uint32_t width) { _width = width; }
  void set_height(uint32_t height) { _height = height; }

 private:
  Coordinate* _coordinate = new Coordinate();
  uint32_t _width = 0;
  uint32_t _height = 0;
};

}  // namespace ipl::imp