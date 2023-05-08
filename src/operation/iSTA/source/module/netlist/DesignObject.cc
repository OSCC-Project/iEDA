/**
 * @file DesignObject.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of base class.
 * @version 0.1
 * @date 2021-02-03
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "DesignObject.hh"
#include "string/Str.hh"

namespace ista {

DesignObject::DesignObject(const char* name) : _name(name) {}

DesignObject::DesignObject(DesignObject&& other) noexcept
    : _name(std::move(other._name)) {}

DesignObject& DesignObject::operator=(DesignObject&& rhs) noexcept {
  _name = std::move(rhs._name);

  return *this;
}

}  // namespace ista
