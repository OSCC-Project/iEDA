#include "gmock/gmock.h"
#include "gtest/gtest-death-test.h"
#include "gtest/gtest.h"
#include "time/Time.hh"

using ieda::Time;

namespace {

TEST(TimeTest, wallTime) {
  const char* now_wall_time = Time::getNowWallTime();

  std::cout << now_wall_time << "\n";
}

}  // namespace
