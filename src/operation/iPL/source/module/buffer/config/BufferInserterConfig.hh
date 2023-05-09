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
 * @Author: sjchanson 13560469332@163.com
 * @Date: 2022-11-18 10:57:27
 * @LastEditors: sjchanson 13560469332@163.com
 * @LastEditTime: 2022-11-29 11:27:52
 * @FilePath: /irefactor/src/operation/iPL/source/module/buffer/config/BufferInserterConfig.hh
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#ifndef IPL_BUFFER_INSERTER_CONFIG_H
#define IPL_BUFFER_INSERTER_CONFIG_H

#include <string>
#include <vector>

namespace ipl {

class BufferInserterConfig{
public:
    BufferInserterConfig() = default;
    BufferInserterConfig(const BufferInserterConfig&) = default;
    BufferInserterConfig(BufferInserterConfig&&) = default;
    ~BufferInserterConfig() = default;

    BufferInserterConfig& operator=(const BufferInserterConfig&) = default;
    BufferInserterConfig& operator=(BufferInserterConfig&&) = default;

    // getter.
    int32_t get_thread_num() const { return _thread_num;}
    bool isMaxLengthOpt() { return (_is_max_length_opt == 1);}
    int32_t get_max_wirelength_constraint() const {return _max_wirelength_constraint;}
    int32_t get_max_buffer_num() const { return _max_buffer_num;}
    std::vector<std::string> get_buffer_master_list() const { return _buffer_master_list;}
        
    // setter.
    void set_thread_num(int32_t num_thread) { _thread_num = num_thread;}
    void set_is_max_length_opt(int32_t flag) { _is_max_length_opt = flag;}
    void set_max_wirelength_constraint(int32_t max_wirelength) { _max_wirelength_constraint = max_wirelength;}
    void set_max_buffer_num(int32_t max_num) { _max_buffer_num = max_num;}
    void add_buffer_master(std::string buffer) { _buffer_master_list.push_back(buffer);}

private:
    int32_t _thread_num;
    int32_t _is_max_length_opt;
    int32_t _max_wirelength_constraint;
    int32_t _max_buffer_num;
    std::vector<std::string> _buffer_master_list;
};



}



#endif