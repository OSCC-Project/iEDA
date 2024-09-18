/*
 * @FilePath: wirelength_lut.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-30 10:44:31
 * @Description:
 */

#pragma once

#include <array>
#include <cstdint>

namespace ieval {
extern const std::array<std::array<std::array<double, 4>, 16>, 4> WIRELENGTH_LUT;
}  // namespace ieval
