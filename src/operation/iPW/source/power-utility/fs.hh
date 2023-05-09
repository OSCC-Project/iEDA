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
// for mmap:
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <ext/stdio_filebuf.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace os::fs {

// ontaining a FILE* from a std::ifstream or a std::ofstream
// credit https://stackoverflow.com/a/19749019/1087626
namespace impl {
using buffer_t = std::basic_ofstream<char>::__filebuf_type;
using io_buffer_t = __gnu_cxx::stdio_filebuf<char>;
inline FILE* cfile(buffer_t* const fb)
{
  return (static_cast<io_buffer_t* const>(fb))  // NOLINT this is NOT a dynamic cast
      ->file();                                 // type std::__c_file
}
}  // namespace impl

inline FILE* cfile(std::ofstream const& ofs)
{
  return impl::cfile(ofs.rdbuf());
}
inline FILE* cfile(std::ifstream const& ifs)
{
  return impl::cfile(ifs.rdbuf());
}

class MemoryMappedFile
{
 public:
  explicit MemoryMappedFile(const std::string& filename)
  {
    int fd = open(filename.c_str(), O_RDONLY);  // NOLINT
    if (fd == -1)
      throw std::logic_error("MemoryMappedFile: couldn't open file.");

    // obtain file size
    struct stat sbuf
    {
    };
    if (fstat(fd, &sbuf) == -1)
      throw std::logic_error("MemoryMappedFile: cannot stat file size");
    filesize_ = static_cast<std::size_t>(sbuf.st_size);

    map_ = static_cast<const char*>(mmap(nullptr, filesize_, PROT_READ, MAP_PRIVATE, fd, 0U));
    if (map_ == MAP_FAILED)  // NOLINT c-style cast in macro + int to ptr cast
                             // pessimisation
      throw std::logic_error("MemoryMappedFile: cannot map file");
  }

  ~MemoryMappedFile()
  {
    if (munmap(static_cast<void*>(const_cast<char*>(map_)), filesize_) == -1)  // NOLINT const_cast
      std::cerr << "Warnng: MemoryMappedFile: error in destructor during "
                   "`munmap()`\n";
  }

  // no copies
  MemoryMappedFile(const MemoryMappedFile& other) = delete;
  MemoryMappedFile& operator=(MemoryMappedFile other) = delete;

  // default moves
  MemoryMappedFile(MemoryMappedFile&& other) = default;
  MemoryMappedFile& operator=(MemoryMappedFile&& other) = default;

  // char* pointers. up to callee to make string_views or strings
  [[nodiscard]] const char* begin() const { return map_; }
  [[nodiscard]] const char* end() const { return map_ + filesize_; }  // NOLINT

  [[nodiscard]] std::string_view get_buffer() const { return std::string_view{begin(), static_cast<std::size_t>(end() - begin())}; }

 private:
  std::size_t filesize_ = 0;
  const char* map_ = nullptr;
};

}  // namespace os::fs