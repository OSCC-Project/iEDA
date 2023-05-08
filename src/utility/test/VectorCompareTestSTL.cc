#include <ctime>
#include <iostream>
#include <vector>
//#include "Vector.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

TEST(vectorCompare, addSTL) {
  absl::Time startTime3, endTime3;
  std::vector<int> ar4;
  startTime3 = absl::Now();
  for (int i = 0; i < 9900000; i++) {
    ar4.push_back(i + 2);
  }
  endTime3 = absl::Now();
  absl::Duration duration = endTime3 - startTime3;
  std::cout << "the STL vector run time is = "
            << duration / absl::Nanoseconds(1) << "ns" << std::endl;
}
