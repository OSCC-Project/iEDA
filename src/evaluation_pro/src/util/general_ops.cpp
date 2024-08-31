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
  char current_path[PATH_MAX];
  if (getcwd(current_path, sizeof(current_path)) != nullptr) {
    std::string working_dir(current_path);

    if (filename[0] == '/') {
      return filename;
    }

    return working_dir + "/" + filename;
  }

  char resolved_path[PATH_MAX];
  if (realpath(filename.c_str(), resolved_path) != nullptr) {
    return std::string(resolved_path);
  }
  return filename;
}
}  // namespace ieval