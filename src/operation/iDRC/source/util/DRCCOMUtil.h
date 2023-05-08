#ifndef IDRC_SRC_UTIL_DRC_COMUTIL_H_
#define IDRC_SRC_UTIL_DRC_COMUTIL_H_

#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <map>
#include <queue>
#include <vector>

namespace idrc {

class DRCCOMUtil
{
 public:
  static double microtime()
  {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec + tv.tv_usec / 1000000.00;
  }

  template <typename T>
  static std::queue<T> initQueue(const T& t)
  {
    std::queue<T> queue;
    queue.push(t);
    return queue;
  }

  template <typename T>
  static T getFrontAndPop(std::queue<T>& queue)
  {
    T node = queue.front();
    queue.pop();
    return node;
  }

  template <typename T>
  static void addListToQueue(std::queue<T>& queue, std::vector<T>& node_list)
  {
    for (size_t i = 0; i < node_list.size(); i++) {
      queue.push(node_list[i]);
    }
  }
};

}  // namespace idrc
#endif  // IDR_SRC_UTIL_COMUTIL_H_
