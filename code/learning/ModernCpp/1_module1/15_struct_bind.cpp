#include <iostream>
#include <vector>

using namespace std;

class Point{
public:

    Point (int x, int y):x(x),y(y){

        cout<<"ctor"<<endl;
    }

    Point(const Point& pt):x(pt.x),y(pt.y)
    {
        cout<<"copy ctor"<<endl;
    }

// private:
    
    int x;
    int y;

};

Point get_point(){

    Point pt{100,200};
    return pt;
}

struct C1{
    int a;
    int b;
    int c;
};

struct C2: C1{

};



int main(){


    Point pt1{300,400};

    auto [x1, y1]=pt1; //拷贝构造
    pt1.x++;
    pt1.y++;
    cout<<x1<<","<<y1<<endl;

    auto [x2, y2]=get_point();
    cout<<x2<<","<<y2<<endl;

    Point pt2{500,600};
    auto& [x3, y3]=pt2; //对象引用
    x3++;
    y3++;
    cout<<pt2.x<<","<<pt2.y<<endl;


    vector<pair<int,int>> prv;
    prv.emplace_back(pair{1,1});
    prv.emplace_back(pair{2,2});
    prv.emplace_back(pair{3,3});
    prv.emplace_back(pair{4,5});
    prv.emplace_back(pair{5,8});
    for(const auto& [k,v]: prv)
    {
        cout<<"{"<<k<<","<<v<<"}"<<endl;
    }

      


    int data[]={1,2,3,4,5};
    auto [x,y,z,u,v]=data;
    cout<<x<<","<<y<<","<<z<<","<<u<<","<<v<<endl;


    C2 c2{10,20,30};
    auto [a,b,c]=c2;
    cout<<a<<","<<b<<","<<c<<endl;


}