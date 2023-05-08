#pragma once

#include <math.h>

#include <iostream>
#include <string>

#include "lef_service.h"

namespace idb {

enum class LefPropertyType
{
  kNone,
  kCutSpacing,
  kMax
};
template <typename T>
class PropertyBaseParser
{
 public:
  explicit PropertyBaseParser(IdbLefService* lef_service) { _lef_service = lef_service; }
  virtual ~PropertyBaseParser() { _lef_service = nullptr; }

  /// operator
  IdbLefService* get_lef_service() { return _lef_service; }
  IdbLayout* get_layout() { return _lef_service != nullptr ? _lef_service->get_layout() : nullptr; }

  int32_t transAreaDB(double value) { return _lef_service->get_layout()->transAreaDB(value); }
  int32_t transUnitDB(double value) { return _lef_service->get_layout()->transUnitDB(value); }

  virtual bool parse(const std::string& name, const std::string& value, T* data) = 0;

 private:
  IdbLefService* _lef_service;
};

}  // namespace idb
