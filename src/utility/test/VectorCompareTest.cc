#include <ctime>
//#include <vector>
#include <iostream>

#include "Vector.hh"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
TEST(VectorCompare, add) {
  absl::Time startTime2, endTime2;
  ieda::Vector<int> ar3;
  startTime2 = absl::Now();
  for (int i = 0; i < 9900000; i++) {
    ar3.push_back(i + 2);
  }
  endTime2 = absl::Now();
  absl::Duration duration = endTime2 - startTime2;
  std::cout << "the vector run time is = " << duration / absl::Nanoseconds(1)
            << "ns" << std::endl;
}
