/*
 * @FilePath: general_ops.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "general_ops.h"

#include <unistd.h>

#include <climits>
#include <cstdlib>
#include <cstring>

namespace ieval {

std::string getAbsoluteFilePath(std::string filename)
{
  char currentPath[PATH_MAX];
  if (getcwd(currentPath, sizeof(currentPath)) != nullptr) {
    std::string workingDir(currentPath);

    if (filename[0] == '/') {
      return filename;
    }

    return workingDir + "/" + filename;
  }

  char resolvedPath[PATH_MAX];
  if (realpath(filename.c_str(), resolvedPath) != nullptr) {
    return std::string(resolvedPath);
  }
  return filename;
}
}  // namespace ieval