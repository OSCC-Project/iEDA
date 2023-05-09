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
#ifndef IDB_MIN_WIDTH_CHECK
#define IDB_MIN_WIDTH_CHECK

namespace idb {
  class IdbMinWidthCheck {
   public:
    IdbMinWidthCheck() { }
    explicit IdbMinWidthCheck(int minWidth) : _min_width(minWidth) { }
    ~IdbMinWidthCheck() = default;

    // getter
    int get_min_width() const { return _min_width; }
    // setter
    void set_min_width(int min_width) { _min_width = min_width; }
    // operator

   private:
    int _min_width;
  };
}  // namespace idb

#endif
