class Point{
public:
	int x;
	int y;
};


class Line{
public:
	Point _start;
    Point _end;

	Line(const Point& start, const Point& end):
		_start(start), _end(end)
	{
 

    }

};

class Rect{
public:
	Point _leftUp;
    int _width;
	int _height;

	Rect(const Point& leftUp, int width, int height):
		_leftUp(leftUp), _width(width), _height(height)
	{

    }

};

//需求变更
class Circle{


};






 

