/*
 * @Author: S.J Chen
 * @Date: 2022-01-21 10:47:12
 * @LastEditTime: 2022-09-08 11:27:03
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/database/Region.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_REGION_H
#define IPL_REGION_H

#include <string>
#include <vector>

#include "Instance.hh"
#include "Rectangle.hh"

namespace ipl {

enum class REGION_TYPE
{
  kNone,
  kFence,
  kGuide
};

class Region
{
 public:
  Region() = delete;
  explicit Region(std::string name) : _name(std::move(name)), _type(REGION_TYPE::kNone) {}
  Region(const Region&) = default;
  Region(Region&& other) noexcept
  {
    _name       = std::move(other._name);
    _boundaries = std::move(other._boundaries);
    _instances  = std::move(other._instances);
    _type       = other._type;

    other._name = "";
    other._boundaries.clear();
    other._instances.clear();
    other._type = REGION_TYPE::kNone;
  }
  ~Region() = default;

  Region& operator=(const Region&) = default;
  Region& operator                 =(Region&& other) noexcept
  {
    _name       = std::move(other._name);
    _boundaries = std::move(other._boundaries);
    _instances  = std::move(other._instances);
    _type       = other._type;

    other._name = "";
    other._boundaries.clear();
    other._instances.clear();
    other._type = REGION_TYPE::kNone;

    return (*this);
  }

  // getter.
  std::string                     get_name() const { return _name; }
  std::vector<Rectangle<int32_t>> get_boundaries() const { return _boundaries; }
  std::vector<Instance*>          get_instances() const { return _instances; }

  

  bool isFence() const { return _type == REGION_TYPE::kFence; }
  bool isGuide() const { return _type == REGION_TYPE::kGuide; }

  // setter.
  void set_type(REGION_TYPE type) { _type = type; }
  void add_boundary(Rectangle<int32_t> boundary) { _boundaries.push_back(std::move(boundary)); }
  void add_instance(Instance* inst) { _instances.push_back(inst); }

 private:
  std::string                     _name;
  std::vector<Rectangle<int32_t>> _boundaries;
  std::vector<Instance*>          _instances;

  REGION_TYPE _type;
};

}  // namespace ipl

#endif