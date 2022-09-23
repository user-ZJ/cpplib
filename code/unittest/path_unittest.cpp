#include<iostream>
#include<string>
#include "path-util.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(PathTest, BasenameTest){
  std::string path = "C:\\Users\\Administrator\\Desktop\\text\\data.22.txt";
  EXPECT_EQ("data.22.txt",path_get_filename(path));
}


TEST(PathTest, SuffixnameTest){
  std::string path = "C:\\Users\\Administrator\\Desktop\\text\\data.22.txt";
  EXPECT_EQ("txt",extension_name(path));
}



// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
