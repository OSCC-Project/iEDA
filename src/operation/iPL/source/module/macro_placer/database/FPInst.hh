
#pragma once

// #include "Utility.h"
// #include "opendb/db.h"

#include <iostream>
#include <string>
#include <vector>

using std::string;
using std::vector;
namespace ipl::imp {

enum class InstType : uint8_t
{
  STD,
  MACRO,
  IO,
  NEWMACRO,
  FLIPFLOP
};
enum class Orient
{
  kNone,
  N,
  E,
  S,
  W,
  FN,
  FE,
  FS,
  FW
};

class Coordinate
{
 public:
  int32_t _x = 0;
  int32_t _y = 0;
};

class FPPin;

class FPInst
{
 public:
  FPInst(){};
  ~FPInst(){};

  // setter
  void set_name(string name) { _name = name; }
  void set_index(int index) { _index = index; }
  void set_type(InstType type) { _type = type; }
  void set_fixed(bool fixed) { _fixed = fixed; }
  void set_width(uint32_t width) { _width = width; }
  void set_height(uint32_t height) { _height = height; }
  void set_halo_x(uint32_t halo) { _halo_x = halo; }
  void set_halo_y(uint32_t halo) { _halo_y = halo; }
  void set_coordinate(Coordinate* coordinate) { _coordinate = coordinate; }
  void set_x(int32_t x) { _coordinate->_x = x; }
  void set_y(int32_t y) { _coordinate->_y = y; }
  void set_orient(Orient orient)
  {
    _orient = orient;
    if (_orient == Orient::N || _orient == Orient::FN || _orient == Orient::S || _orient == Orient::FS) {
      _main_orient = true;
    } else {
      _main_orient = false;
    }
  }
  void add_pin(FPPin* pin) { _pin_list.emplace_back(pin); }
  void pop_pin_list() { _pin_list.pop_back(); }
  void set_align_flag(bool flag) { _align_flag = flag; }

  // getter
  string get_name() const { return _name; }
  int get_index() const { return _index; }
  int32_t get_x() const { return _coordinate->_x; }
  int32_t get_y() const { return _coordinate->_y; }
  uint32_t get_width() const
  {
    if (_main_orient) {
      return _width;
    } else {
      return _height;
    }
  }
  uint32_t get_height() const
  {
    if (_main_orient) {
      return _height;
    } else {
      return _width;
    }
  }
  uint32_t get_halo_x() const { return _halo_x; }
  uint32_t get_halo_y() const { return _halo_y; }
  float get_area() const { return float(_width) * float(_height); }
  Coordinate* get_coordinate() const { return _coordinate; }
  InstType get_type() const { return _type; }
  Orient get_orient() const { return _orient; }
  // Different directions, different widths and heights
  int32_t get_center_x() { return _coordinate->_x + get_width() * 0.5; }
  int32_t get_center_y() { return _coordinate->_y + get_height() * 0.5; }
  vector<FPPin*> get_pin_list() { return _pin_list; }
  bool isFixed() { return _fixed; }
  void addHalo()
  {
    if (!_has_halo) {
      _width += 2 * _halo_x;
      _height += 2 * _halo_y;
      _coordinate->_x -= _halo_x;
      _coordinate->_y -= _halo_y;
      _has_halo = true;
    }
  }
  void deleteHalo()
  {
    if (_has_halo) {
      _width -= 2 * _halo_x;
      _height -= 2 * _halo_y;
      _coordinate->_x += _halo_x;
      _coordinate->_y += _halo_y;
      _has_halo = false;
    }
  }
  bool isMacro() { return _type == InstType::MACRO; }
  string get_orient_name()
  {
    switch (_orient) {
      case Orient::N:
        return "N,R0";
      case Orient::S:
        return "S,R180";
      case Orient::W:
        return "W,R90";
      case Orient::E:
        return "E,R270";
      case Orient::FN:
        return "FN,MY";
      case Orient::FS:
        return "FS,MX";
      case Orient::FW:
        return "FW,MX90";
      case Orient::FE:
        return "FE,MY90";

      default:
        return "kNone,kNone";
        break;
    }
  }
  bool isAlign() { return _align_flag; }

 private:
  string _name;
  int _index;
  InstType _type;
  bool _fixed = false;
  uint32_t _width;
  uint32_t _height;
  uint32_t _halo_x = 0;
  uint32_t _halo_y = 0;
  bool _has_halo = false;
  Coordinate* _coordinate = new Coordinate();
  Orient _orient = Orient::N;
  bool _main_orient = true;
  vector<FPPin*> _pin_list;
  bool _align_flag = false;
};
}  // namespace ipl::imp