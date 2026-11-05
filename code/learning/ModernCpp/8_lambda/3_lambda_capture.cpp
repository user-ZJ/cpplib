
#include <iostream>
#include <vector>

using namespace std;

class Point{
public:
    double x;
    double y;
    void print() const {
        std::cout << x<< ", "<<y<<endl;;
    }
};


/*
struct Lambda_Ref{

    Point& p1; //8byte
    Point& p2; //8byte

    Lambda_Ref( Point& p1, Point& p2):p1(p1),p2(p2)
    {

    }
    void operator()(int n) {
        p1.x+=n;
        p1.y+=n;
        p2.x+=n;
        p2.y+=n;
    }
};*/


/*
struct Lambda_Value{

    Point p1;
    Point p2;

    Lambda_Value(const Point& p1, const Point& p2):p1(p1),p2(p2)
    {

    }
    void operator()(int n) {
        p1.x+=n;
        p1.y+=n;
        p2.x+=n;
        p2.y+=n;

        p1.print();
        p2.print();

        number++;
    }
};
*/

int number=100;




int main()
{
	Point p1{100,200};
    Point p2{100,200};

  
    auto lambda1 = [=] (int n) mutable    //Lambda_Value lambda1(p1,p2);
	{
		p1.x+=n;
        p1.y+=n;
        p2.x+=n;
        p2.y+=n;
        p1.print();
        p2.print();

        number++;

	};

    

     lambda1(10);

     lambda1(10);
     p1.print();
     p2.print();
     cout<<sizeof(lambda1)<<endl;
     cout<<sizeof(Point)<<endl;
     cout<<number<<endl;
    
	cout<<"lambda1------------"<<endl;

	auto lambda2 = [&] (int n)
	{
        p1.print();
        p2.print();
		p1.x+=n;
        p1.y+=n;
        p2.x+=n;
        p2.y+=n;

        number++;
	};

    //Lambda_Ref lambda2(p1,p2);

    {
        lambda2(100);
    
        p1.print();
        p2.print();

        p1.x+=5;
        p1.y+=5;
        p2.x+=5;
        p2.y+=5;
        lambda2(100);
    }

    
    cout<<number<<endl;
    cout<<sizeof(lambda2)<<endl;


    cout<<"lambda2------------"<<endl;

    auto lambda3 = [=, &p1] ()
	{
		p1.x++;
        p1.y++;
        p2.print();
	};

     auto lambda4 = [ &, p1] ()
	{
        p1.print();
		p2.x++;
        p2.y++;
        
	};

    


    auto lambda5 = [p2] ()
	{
        p2.print();
	};

    auto lambda6 = [&p1] ()
	{
		p1.x++;
        p1.y++;
	};

    auto lambda7 = [p1,&p2] ()
    {
        p2.x=p1.y;
        p2.y=p1.x;
	};

    cout<<sizeof(lambda7)<<endl;
}
