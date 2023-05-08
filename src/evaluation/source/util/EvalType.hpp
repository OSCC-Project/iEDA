#ifndef SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALTYPE_HPP_
#define SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALTYPE_HPP_

namespace eval {
enum class NET_TYPE
{
  kNone,
  kSignal,
  kClock,
  kReset,
  kFakeNet
};

enum class PIN_TYPE
{
  kNone,
  kInstancePort,
  kIOPort,
  kFakePin
};

enum class PIN_IO_TYPE
{
  kNone,
  kInput,
  kOutput,
  kInputOutput
};

enum class INSTANCE_TYPE
{
  kNone,
  kNormal,
  kOutside,
  kFakeInstance
};

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_UTIL_COMMON_EVALTYPE_HPP_
