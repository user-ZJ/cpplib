
#include <iostream>
using namespace std;


/*
template<typename Mixins>
class Point : public Mixins
{
public:
    double x, y;

    Point() : Mixins(), x(0.0), y(0.0) { }
    Point(double x, double y) : Mixins(), x(x), y(y) { }
};*/




template<typename... Mixins>
class Point : public Mixins...
{
public:
    double x, y;

    Point() : Mixins()..., x(0.0), y(0.0) { }
    Point(double x, double y) : Mixins()..., x(x), y(y) { }
};



template<typename... Mixins>
class Line : public Mixins...
{
public:
    double x1, y1,x2,y2 ;

    Line() : Mixins()..., x1(0.0), y1(0.0), x2(0),y2(0) { }
    Line(double x1, double y1, double x2, double y2) : Mixins()..., x1(x1), y1(y1),x2(x2),y2(y2) { }
};


/*

n=5
m=5

5+5* 15= 80
5+5=10

n+n*(m+m-1....1)
n+m



class Point{};
class Line{};


class LabelPoint: public Point{};
class LableLine: public Line{};

class ColorPoint: public Point{};
class ColorLine: public Line{};

class LabelColorPoint: ColorPoint{};
class LabelColorLine: ColorLine{};
*/

class Label
{
public:
    std::string label;
    Label() : label("") { }
};

class Color
{
public:
    unsigned char red = 0, green = 0, blue = 0;
};



using LabelPoint = Point<Label>;
using ColorPoint = Point<Color>;
using LabelColorPoint = Point<Label, Color>;

int main()
{
    LabelColorPoint pt;
    pt.x=100;
    pt.y=200;
    pt.label="2D";
    pt.blue=255;
    pt.red=255;
    pt.green=255;


}





