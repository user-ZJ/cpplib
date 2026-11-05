
#include <iostream>
#include <vector>

using namespace std;


// struct lambda{
//     Point* this;
//    

//     void operator ()(){
//             this->x++;
//             this->y++;
//           
     
//     };

// };

class Point{
public:
    double x,y;

    auto invoke()
    {
        auto lamb = [this] ()
        {
            
            x++;
            y++;
            cout<<x<<','<<y<<endl;
        };
        return lamb;
    }
};





auto process()
{
    Point p1{100,200};
    auto lambda=p1.invoke(); 
    return lambda; // return &p1;
}

int main(){

    auto lam=process();

    lam();
    
}


