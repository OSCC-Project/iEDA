#include "ToApi.hpp"

int main() {
  std::string config_file = " ";

  ToApiInst.initTO(config_file);
  ToApiInst.iTODataInit();
  ToApiInst.runTO();

  ToApiInst.destroyInst();
  return 0;
}