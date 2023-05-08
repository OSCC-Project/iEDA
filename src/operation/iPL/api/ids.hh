/*
 * @Author: S.J Chen
 * @Date: 2022-10-27 14:50:40
 * @LastEditors: sjchanson 13560469332@163.com
 * @LastEditTime: 2022-12-14 18:51:00
 * @FilePath: /irefactor/src/operation/iPL/api/ids.hh
 * @Description:
 */

#ifndef IPL_IDS_H
#define IPL_IDS_H

#include <any>
#include <map>
#include <string>
#include <vector>

namespace ids {

}

namespace idb {

class IdbBuilder;

enum class IdbConnectType : uint8_t;
}  // namespace idb

namespace ipl {
class NetWork;

template <typename T>
class Rectangle;

template <typename T>
class Point;

}  // namespace ipl

namespace eval {
class EvalAPI;
class TimingPin;
class CongGrid;
class CongInst;

}  // namespace eval

namespace ista {

enum class AnalysisMode;

}

#endif  // SRC_OPERATION_IPL_API_IDS_HH_
