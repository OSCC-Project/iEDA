/**
 * @file Time.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The utility class of time.
 * @version 0.1
 * @date 2021-05-08
 */
#pragma once

namespace ieda {

/**
 * @brief The utility class relate to the time.
 *
 */
class Time
{
 public:
  static const char* getNowWallTime();
};

}  // namespace ieda