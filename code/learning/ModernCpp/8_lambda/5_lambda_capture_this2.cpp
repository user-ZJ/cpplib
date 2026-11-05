
#include <iostream>
#include <vector>

using namespace std;


// struct lambda{
//     Point this;


//     void operator ()(){
//             this->x++;
//             this->y++;

     
//     };

// };

class Point{
public:
    double x;
    double y;


    void print() const{
        std::cout << x<< ", "<<y<<endl;;
    }

    auto invoke()
    {
        // auto lambda = [*this] () mutable
        // {
        //     x++;
        //     (*this).x++;
        //     this->x++;
        //     this->y++;
        //     this->print();
            
        // };

        auto lambda = [pt=*this] () mutable
        {
            pt.x++;
            pt.y++;
            pt.print();
            
        };

        return lambda;

    }
};


int main()
{
	Point p1{100,200};
    auto lambda=p1.invoke();// invoke(&p1);
    
    lambda();
    cout<<sizeof(lambda)<<endl;
    p1.print();



	
}