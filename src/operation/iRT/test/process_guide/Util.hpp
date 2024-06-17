template <typename T>
T* getFileStream(std::string file_path)
{
  T* file = new T(file_path);
  if (!file->is_open()) {
    std::cout << "Failed to open file '" << file_path << "'!" << std::endl;
    assert(0);
  }
  return file;
}

std::ifstream* getInputFileStream(std::string file_path)
{
  return getFileStream<std::ifstream>(file_path);
}

std::ofstream* getOutputFileStream(std::string file_path)
{
  return getFileStream<std::ofstream>(file_path);
}

template <typename T>
static void closeFileStream(T* t)
{
  if (t != nullptr) {
    t->close();
    delete t;
  }
}

int32_t countDifferentLines(const std::string& file1, const std::string& file2)
{
  std::ifstream fileStream1(file1);
  std::ifstream fileStream2(file2);
  std::string line1, line2;
  int32_t count = 0;

  while (std::getline(fileStream1, line1) && std::getline(fileStream2, line2)) {
    if (line1 != line2) {
      count++;
    }
  }

  while (std::getline(fileStream1, line1)) {
    count++;
  }

  while (std::getline(fileStream2, line2)) {
    count++;
  }

  fileStream1.close();
  fileStream2.close();

  return count;
}