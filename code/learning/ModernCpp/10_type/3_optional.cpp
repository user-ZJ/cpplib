#include <string>
#include <functional>
#include <iostream>
#include <optional>

using namespace std;
 

optional<string> create1(bool flag) {
    if (flag)
        return "C++ Language";
    return {};
}
 

auto create2(bool flag) {
    return flag ? optional<string>{"C Language"} : nullopt;
}
 

auto create_ref(bool flag) {
    static string value = "Rust Language";
    return flag ? optional<reference_wrapper<string>>{value}
             : nullopt;
}
 
int main()
{
      if (optional<string> str = create1(true)) {
        cout  << *str << '\n';
        cout<<sizeof(str)<<endl;
    }
 

    optional<string> value=create2(false);
    cout<<sizeof(value)<<endl;
    cout << value.value_or("default") << '\n';

  
    if (auto str = create_ref(true)) {
        cout  << str->get() << '\n';
        str->get() = "Carbon";
        cout  << str->get() << '\n';
    }
}