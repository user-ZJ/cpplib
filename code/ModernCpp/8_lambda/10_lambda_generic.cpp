#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;



int main()
{


    auto compareLam=[](auto x, auto y) -> bool { return x < y; };
    auto printLam=[](auto item){ cout<<item<<" ";};

    vector v1 = { 7,2,8,4,3 };
    sort(v1.begin(),v1.end(), compareLam);  
    for_each(v1.begin(),v1.end(), printLam);

    cout<<endl;

 
    vector<string> v2 = { "Python"s,"C"s,"Java"s,"C++"s,"GO"s,"Rust"s };
  
    sort(v2.begin(),v2.end(), compareLam);   
    for_each(v2.begin(),v2.end(), printLam);
    cout<<endl;
  
    int d1=234;
    double d2=342.24;
    cout<<std::boolalpha<<compareLam(d1,d2)<<endl;
    
}
