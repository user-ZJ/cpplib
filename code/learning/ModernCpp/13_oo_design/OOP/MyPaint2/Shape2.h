class Shape{
public:
	virtual void Draw(const Graphics& g)=0;
	virtual ~Shape() { }
};



class Point{
public:
	int x;
	int y;

};


class Line: public Shape 
{
public:
	Point _start;
    Point _end;

	Line(const Point& start, const Point& end):
		_start(start), _end(end)
	{
    }

	void Draw(const Graphics& g) override
	{
		g.DrawLine(Pens.Red, 
			_start.x, _start.y,_end.x, _end.y);
	}

};

class Rect: public Shape{
public:
	Point _leftUp;
    int _width;
	int _height;

	Rect(const Point& leftUp, int width, int height):
		_leftUp(leftUp), _width(width), _height(height)
	{

    }

	void Draw(const Graphics& g) override
	{
		g.DrawRectangle(Pens.Red,
			_leftUp,_width,_height);
	}

};




//扩展
class Circle: public Shape{
public:

	void Draw(const Graphics& g) override
	{
		//....
	}
};






