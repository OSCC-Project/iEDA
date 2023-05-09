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
#ifndef IDB_MIN_AREA_CHECK
#define IDB_MIN_AREA_CHECK

namespace idb {
  class IdbMinAreaCheck {
   public:
    IdbMinAreaCheck() { }
    explicit IdbMinAreaCheck(int min_area) : _min_area(min_area) { }
    ~IdbMinAreaCheck() = default;
    // getter
    int get_min_area() const { return _min_area; }
    // setter
    void set_min_area(int min_area) { _min_area = min_area; }
    // operator

   private:
    int _min_area;
  };
}  // namespace idb

#endif
