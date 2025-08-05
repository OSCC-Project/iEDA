/*
 * @FilePath: general_ops.h
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#pragma once

#include <string>

namespace ieval {

std::string getAbsoluteFilePath(std::string filename);
std::string createDirPath(std::string dir_path);
std::string getDefaultOutputPath();
bool createDirectoryRecursive(const std::string& path);

}  // namespace ieval
