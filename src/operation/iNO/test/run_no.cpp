#include "NoApi.hpp"

int main() {
  std::string config_file = " ";

  NoApiInst.initNO(config_file);
  NoApiInst.iNODataInit(nullptr, nullptr);
  NoApiInst.fixFanout();

  NoApiInst.destroyInst();
  return 0;
}