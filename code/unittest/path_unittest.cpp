#include<iostream>
#include<string>
#include "path-util.h"
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
  EXPECT_EQ(R"(C:\Users\Administrator\..\..\Desktop\text\data.22.txt/a.txt)",path_combine(path,"a.txt"));
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






// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
