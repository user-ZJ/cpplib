#include<iostream>
#include<filesystem>

namespace fs = std::filesystem;
using namespace std;

int test1(){

    fs::path path_1("/home/wei/ubuntu/Test/c17_filesystem/CMakeLists.txt");
    cout<<"path:"<<path_1.string()<<endl;

    // 判断文件
    cout << "is_regular_file:"<<fs::is_regular_file(path_1) << endl;
    // 判断路径
    cout << "is_directory:"<<fs::is_directory(path_1) << endl;

    //  判断是否存在
    cout <<"exists:"<< fs::exists(path_1) << endl;

    // Path 的属性
    cout << "filename:"<<path_1.filename() << endl;              // 包含后缀, CMakeLists.txt
    cout << "root_directory:"<<path_1.root_directory() << endl;
    cout << "root_path:"<<path_1.root_path() << endl;
    cout << "parent_path:"<<path_1.parent_path() << endl;
    cout << "extension:"<<path_1.extension() << endl;
    cout << "is_absolute:"<<path_1.is_absolute() << endl;
    cout << "is_relative:"<<path_1.is_relative() << endl;
    cout << "stem:"<<path_1.stem() << endl;                  // 不包含后缀的文件名, CMakeLists
    
    return 0;
}

void test2() {

    auto my_path = fs::path("/home/usr");

    // 遍历路径, "/", "home", "usr"
    for(auto& item : my_path) {
        cout << "item:"<<item.string() << endl;
    }

}

void test3() {
    auto my_path = fs::path("/home/wei");
    // ==========修改路径============================

    // my_path.append 类似于python os.path.join()
    my_path.append("Software");
    my_path.concat("QQ");       // my_path.concat() 只是简单的拼接字符串,没有添加目录分割符
    cout << my_path.string() << endl;
    my_path.clear();
    my_path.assign("/tmp/aaa.txt");     // 赋值新的路径
    cout << my_path.remove_filename().string() << endl;     //删除文件名
    cout << my_path.assign("/tmp/aaa.txt").replace_filename("bbb").string() << endl;        // 替换修改文件名
    cout << my_path.assign("/tmp/aaa.txt").replace_extension(".cpp").string() << endl;        // 删除扩展名

    // ==========路径比较=======================
    // cout << fs::path("/tmp/abc.txt") == fs::path("/tmp/abb.txt") << endl;
    cout << fs::path("/tmp/abc.txt").compare("/tmp") << endl;
}

int main(){
    test1();
    test2();
    test3();
}