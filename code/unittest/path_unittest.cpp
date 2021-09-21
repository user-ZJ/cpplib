#include<iostream>
#include<string>
#include "path.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(PathTest, BasenameTest1){
  std::string path = "C:\\Users\\Administrator\\Desktop\\text\\data.22.txt";
  EXPECT_EQ("data.22",basename(path,false));
}

TEST(PathTest, BasenameTest2){
  std::string path = "C:\\Users\\Administrator\\Desktop\\text\\data.22.txt";
  EXPECT_EQ("data.22.txt",basename(path,true));
}


TEST(PathTest, SuffixnameTest){
  std::string path = "C:\\Users\\Administrator\\Desktop\\text\\data.22.txt";
  EXPECT_EQ("txt",suffixname(path));
}



// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
