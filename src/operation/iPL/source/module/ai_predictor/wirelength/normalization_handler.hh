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
#ifndef NORMALIZATION_HANDLER_HH
#define NORMALIZATION_HANDLER_HH

#include <vector>
#include <string>
#include <memory>

namespace ipl {

class NormalizationHandler {
public:
    NormalizationHandler();
    ~NormalizationHandler();

    // Load MinMaxScaler parameters from JSON file
    bool loadMinMaxParams(const std::string& params_path);

    // Set MinMaxScaler parameters manually
    void setMinMaxParams(const std::vector<float>& data_min, 
                        const std::vector<float>& data_max,
                        const std::vector<std::string>& feature_names = {});

    // Normalize input features using MinMax scaling
    std::vector<float> normalize(const std::vector<float>& features) const;

    // Check if normalization parameters are loaded
    bool isReady() const;

    // Get feature names 
    std::vector<std::string> getFeatureNames() const;

    // Get number of features
    size_t getFeatureCount() const;

private:
    std::vector<float> _data_min;
    std::vector<float> _data_max;
    std::vector<std::string> _feature_names;
    bool _is_loaded = false;

    // Parse JSON file to extract normalization parameters
    bool _parseJsonParams(const std::string& params_path);
};

} // namespace ipl

#endif // NORMALIZATION_HANDLER_HH
