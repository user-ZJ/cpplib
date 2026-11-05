#include <iostream>
#include <vector>

using namespace std;

struct Point{
    double x;
    double y;
};

// template <typename T>
// void process(T t) {
//     cout<<" sizeof T <= 8  "<<endl; 
// }

// template <typename T>
// void process(const T& t) {
//     cout<<" sizeof T > 8  "<<endl;
// }

template <typename T , std::enable_if_t< sizeof(T)<=8 > * = nullptr>
void process(T t) {
    cout<<" sizeof T <= 8  "<<endl;
}

template <typename T, std::enable_if_t<  (sizeof(T)>8) >* = nullptr>
void process(const T& t) {
    cout<<" sizeof T > 8  "<<endl;
}

int main(){

    int data=100;
    double pai=3.1415;
    string text="hello";
    Point point{10.1,20.2};
    Point* pt=new Point{10.1,20.2};

    process(data);
    process(pai);
    process(pt);
    process(text);
    process(point);


}