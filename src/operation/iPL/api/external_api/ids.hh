// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
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

class TopologyManager;
}  // namespace ipl

namespace ieval {
// wirelength
struct TotalWLSummary;
// timing
struct TimingNet;
struct TimingPin;
// congestion
struct OverflowSummary;

}  // namespace ieval

namespace ista {
enum class AnalysisMode;
}

namespace ieda {
class ReportTable;
}

#endif  // SRC_OPERATION_IPL_API_IDS_HH_
