#include <utility>
#include <string>
#include <iostream>
#include <type_traits>

using namespace std;

template<typename T>
using EnableIfString = enable_if_t<is_convertible_v<T,std::string>>;

class Widget
{
  private:
    std::string name;
  public:

    template<typename T,typename U= enable_if_t<is_convertible_v<T,std::string>> >
    explicit Widget(T&& n): name(std::forward<T>(n)) {
        std::cout << "Widget(T&& n) " << name << "\n";
    }

    // template<typename T>
    // explicit Widget(T&& n): name(std::forward<T>(n)) {
    //     std::cout << "Widget(T&& n) " << name << "\n";
    // }

    Widget (Widget const& p) : name(p.name) {
        std::cout << "Widget (Widget const& p) " << name << "\n";
    }
    Widget (Widget&& p) : name(std::move(p.name)) {
        std::cout << "Widget (Widget&& p) " << name << "\n";
    }
};


int main()
{
  string text = "C++"s;
  Widget p1(text);              
  Widget p2("Carbon");    

  Widget p3(p1);            
  Widget p4(std::move(p1)); 


  
}

