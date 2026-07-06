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
#pragma once

#include "Types.hh"

#include <optional>
#include <utility>

namespace ircx {

struct ResistanceTemperatureCoefficients
{
  F64 crt1 = 0.0;
  F64 crt2 = 0.0;

  auto empty() const -> bool { return crt1 == 0.0 && crt2 == 0.0; }
};

inline F64 resistanceTemperatureDeratingFactor(F64 operating_temperature, F64 nominal_temperature, F64 crt1, F64 crt2)
{
  const F64 delta_temperature = operating_temperature - nominal_temperature;
  return 1.0 + crt1 * delta_temperature + crt2 * delta_temperature * delta_temperature;
}

template <typename Layer, typename Corner>
inline auto resistanceNominalTemperature(const Layer& layer, const Corner& corner) -> F64
{
  return layer.has_t0() ? static_cast<F64>(layer.get_t0()) : static_cast<F64>(corner.get_global_temperature());
}

inline F64 applyResistanceTemperatureDerating(F64 base_resistance, F64 operating_temperature, F64 nominal_temperature, F64 crt1, F64 crt2)
{
  return base_resistance * resistanceTemperatureDeratingFactor(operating_temperature, nominal_temperature, crt1, crt2);
}

inline F64 applyResistanceTemperatureDerating(F64 base_resistance, F64 operating_temperature, F64 nominal_temperature,
                                              ResistanceTemperatureCoefficients coefficients)
{
  if (coefficients.empty()) {
    return base_resistance;
  }

  return applyResistanceTemperatureDerating(base_resistance, operating_temperature, nominal_temperature, coefficients.crt1,
                                            coefficients.crt2);
}

template <typename Layer, typename Query>
inline auto resistanceTemperatureCoefficients(const Layer& layer, Query&& query) -> ResistanceTemperatureCoefficients
{
  ResistanceTemperatureCoefficients coefficients;

  std::optional<double> query_crt1;
  std::optional<double> query_crt2;
  std::forward<Query>(query)(query_crt1, query_crt2);

  if (query_crt1.has_value()) {
    coefficients.crt1 = query_crt1.value();
  } else if (auto crt1 = layer.get_crt1()) {
    coefficients.crt1 = crt1.value();
  }

  if (query_crt2.has_value()) {
    coefficients.crt2 = query_crt2.value();
  } else if (auto crt2 = layer.get_crt2()) {
    coefficients.crt2 = crt2.value();
  }

  return coefficients;
}

}  // namespace ircx
