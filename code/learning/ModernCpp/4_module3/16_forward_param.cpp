#include <iostream>
#include <vector>
using namespace std;


/*
class Widget {
public:

	Widget(string& n, vector<int>& c):
        name(n), coordinates(c)
    {}

    Widget(string&& n, vector<int>&& c):
        name(std::move(n)), 
        coordinates(std::move(c))
    {

    }

    Widget(string& n, vector<int>&& c):
        name(n), 
        coordinates(std::move(c))
    {

    }

    Widget(string&& n, vector<int>& c):
        name(std::move(n)), 
        coordinates(c)
    {

    }
private:
	std::string name;
	std::vector<int> coordinates;
}; 
*/

class Widget {
public:
	template<typename T1, typename T2>
	Widget(T1&& n, T2&& c) 		// n 和 c可以绑定任何值 
	: name(std::forward<T1>(n)), 	// 将 n 转发给string构造 
	coordinates(std::forward<T2>(c)) // 将 c 转发给vector构造 
	{}
private:
	std::string name;
	std::vector<int> coordinates;
}; 

int main(){
    
}


