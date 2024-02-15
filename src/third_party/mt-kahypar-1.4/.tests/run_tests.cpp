#include <gmock/gmock.h>
#include <thread>

#include "mt-kahypar/definitions.h"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  mt_kahypar::TBBInitializer::instance(std::thread::hardware_concurrency());
  const int result = RUN_ALL_TESTS();
  mt_kahypar::TBBInitializer::instance().terminate();

  return result;
}