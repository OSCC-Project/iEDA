#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace iplf {

#define iTOInst (ToIO::getInstance())
class ToIO
{
 public:
  static ToIO* getInstance()
  {
    if (!_instance) {
      _instance = new ToIO;
    }
    return _instance;
  }

  /// io
  bool runTO(std::string config = "");
  bool runTOFixFanout(std::string config = "");
  bool runTODrv(std::string config = "");
  bool runTOHold(std::string config = "");
  bool runTOSetup(std::string config = "");

 private:
  static ToIO* _instance;

  ToIO() {}
  ~ToIO() = default;

  void resetConfig();
};

}  // namespace iplf
