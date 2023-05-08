#pragma once

#include <time.h>

namespace idb {

// creation time and the last modification are now by default
class GdsTimestamp
{
 public:
  GdsTimestamp();
  explicit GdsTimestamp(time_t b, time_t l);

  void reset();

  time_t beg;
  time_t last;
};

/////////////// inline ///////////

inline GdsTimestamp::GdsTimestamp() : beg(time(nullptr)), last(time(nullptr))
{
}

inline GdsTimestamp::GdsTimestamp(time_t b, time_t l) : beg(b), last(l)
{
}

inline void GdsTimestamp::reset()
{
  beg = time(nullptr);
  last = time(nullptr);
}

}  // namespace idb