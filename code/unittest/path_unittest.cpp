#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include "utils/path-util.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

std::string path = "C:\\Users\\Administrator\\..\\..\\Desktop\\text\\data.22.txt";

TEST(PathTest, path_append_slash){
  EXPECT_EQ(R"(C:\Users\Administrator\..\..\Desktop\text\data.22.txt/)",path_append_slash(path,'/'));
}

TEST(PathTest, path_remove_slash){
  EXPECT_EQ(R"(C:\Users\Administrator\..\..\Desktop\text\data.22.txt)",path_remove_slash(path_append_slash(path,'/')));
}

TEST(PathTest, path_has_slash){
  EXPECT_EQ(false,path_has_slash(path));
}

TEST(PathTest, path_convert_slashes){
  EXPECT_EQ(R"(C:/Users/Administrator/../../Desktop/text/data.22.txt)",path_convert_slashes(path,'/'));
}

TEST(PathTest, path_remove_filename){
  EXPECT_EQ(R"(C:\Users\Administrator\..\..\Desktop\text)",path_remove_filename(path));
}

TEST(PathTest, path_remove_extension){
  EXPECT_EQ(R"(C:\Users\Administrator\..\..\Desktop\text\data.22)",path_remove_extension(path));
}

TEST(PathTest, path_combine){
  EXPECT_EQ(std::string(R"(C:\Users\Administrator\..\..\Desktop\text\data.22.txt)") + PATH_SLASH_PLATFORM + "a.txt",
            path_combine(path, "a.txt"));
}

TEST(PathTest, extension_name){
  // std::string path = "C:\\Users\\Administrator\\Desktop\\text\\data.22.txt";
  EXPECT_EQ("txt",extension_name(path));
}

TEST(PathTest, path_get_filename){
  // std::string path = "C:\\Users\\Administrator\\Desktop\\text\\data.22.txt";
  EXPECT_EQ("data.22.txt",path_get_filename(path));
}

TEST(PathTest, path_resolve){
  EXPECT_EQ(R"(C:\Desktop\text\data.22.txt)",path_resolve(path));
}

TEST(PathTest, dirMakeAndIsDir) {
  auto root = std::filesystem::temp_directory_path() / ("mbase_ut_path_" + std::to_string(getpid()));
  std::filesystem::remove_all(root);
  auto nested = root / "a" / "b";
  EXPECT_EQ(0, dir_make(nested.string().c_str()));
  EXPECT_TRUE(is_dir(root.string().c_str()));
  EXPECT_TRUE(is_dir(nested.string().c_str()));
  EXPECT_EQ(0, dir_make(nested.string().c_str()));
  EXPECT_FALSE(is_dir((root / "missing").string().c_str()));

  auto file = root / "f.txt";
  std::ofstream(file.string()) << "x";
  EXPECT_FALSE(is_dir(file.string().c_str()));
  EXPECT_FALSE(is_symlink(nested.string().c_str()));
  std::filesystem::remove_all(root);
}






// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
