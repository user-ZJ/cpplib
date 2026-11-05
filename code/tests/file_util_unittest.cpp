#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>
#include "utils/file-util.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

class FileUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() / ("mbase_ut_file_" + std::to_string(getpid()));
    std::filesystem::create_directories(dir_);
  }
  void TearDown() override { std::filesystem::remove_all(dir_); }

  std::string Path(const char *name) const { return (dir_ / name).string(); }

  std::filesystem::path dir_;
};

TEST_F(FileUtilTest, missingFile) {
  auto missing = Path("no-such-file");
  EXPECT_FALSE(is_exist(missing.c_str()));
  EXPECT_EQ(0, get_file_size(missing.c_str()));
  EXPECT_TRUE(file_to_buff(missing.c_str()).empty());
  EXPECT_TRUE(file_to_str(missing.c_str()).empty());
  EXPECT_TRUE(get_file_lines(missing.c_str()).empty());
}

TEST_F(FileUtilTest, linesAndText) {
  auto path = Path("lines.txt");
  EXPECT_EQ(0, write_lines_file(path.c_str(), {"a", "b"}));
  auto lines = get_file_lines(path.c_str());
  ASSERT_EQ(2u, lines.size());
  EXPECT_EQ("a", lines[0]);
  EXPECT_EQ("b", lines[1]);
  EXPECT_EQ(file_to_str(path.c_str()), file_to_ss(path.c_str()).str());

  EXPECT_EQ(1, write_lines_file(Path("missing_dir/x.txt").c_str(), {"a"}));

  auto numbers = Path("numbers.txt");
  EXPECT_EQ(0, writeTextFile(numbers.c_str(), std::vector<int>{1, 2, 3, 4}, 2));
  std::vector<int> read_back;
  EXPECT_EQ(0, readTextFile(numbers, read_back));
  EXPECT_EQ((std::vector<int>{1, 2, 3, 4}), read_back);
  EXPECT_EQ(0, writeTextFile(Path("note.txt").c_str(), std::string("hello")));
  EXPECT_EQ("hello", file_to_str(Path("note.txt").c_str()));
}

TEST_F(FileUtilTest, binaryAndSize) {
  auto path = Path("bin.dat");
  EXPECT_EQ(0, writeBinaryFile(path.c_str(), std::string("abc")));
  EXPECT_TRUE(is_exist(path.c_str()));
  EXPECT_EQ(3, get_file_size(path.c_str()));
  EXPECT_EQ(0, get_file_size(dir_.string().c_str()));
  auto buff = file_to_buff(path.c_str());
  EXPECT_EQ(std::string("abc"), std::string(buff.begin(), buff.end()));

  std::vector<char> values{'a', 'b', 'c', 'd'};
  auto ivec = Path("chars.bin");
  EXPECT_EQ(0, writeBinaryFile(ivec.c_str(), values));
  std::vector<char> loaded;
  EXPECT_EQ(0, readBinaryFile(ivec.c_str(), loaded));
  EXPECT_EQ(values, loaded);

  MappedFile mapped(path);
  ASSERT_EQ(3u, mapped.size);
  EXPECT_EQ(0, std::memcmp(mapped.data, "abc", 3));
}
