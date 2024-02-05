#include <iostream>
#include <boost/functional/hash.hpp>

int main() {
    std::string str = "gpt-3.5-turbo-16k-0613";
    std::size_t hash = boost::hash_value(str);
    std::cout << "Hash value of \"" << str << "\" is: " << hash << std::endl;
    return 0;
}