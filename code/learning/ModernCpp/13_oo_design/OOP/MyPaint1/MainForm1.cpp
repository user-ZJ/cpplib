#include <iostream>
#include <vector>
#include "Shape1.h"
using namespace std;

class Form{};

class MainForm : public Form {
private:
	Point p1;
	Point p2;

	std::vector<Line> lines;
	std::vector<Rect> rects;

	//更改
	std::vector<Circle> circles;

	
public:
	MainForm(){
		//...
	}
protected:


	void MainForm::OnMouseDown(const MouseEventArgs& e) override {
		p1.x = e.X;
		p1.y = e.Y;

		//...
		Form::OnMouseDown(e);
	}

	void MainForm::OnMouseUp(const MouseEventArgs& e) override {
		p2.x = e.X;
		p2.y = e.Y;

		if (rdoLine.Checked){
			Line line(p1, p2);
			lines.push_back(line);
		}
		else if (rdoRect.Checked){
			int width = abs(p2.x - p1.x);
			int height = abs(p2.y - p1.y);
			Rect rect(p1, width, height);
			rects.push_back(rect);
		}
		// 更改....
		else if(...)
		{

		}
		


		//...
		this->Refresh();

		Form::OnMouseUp(e);
	}

	void MainForm::OnPaint(const PaintEventArgs& e) override {

		for (auto& line : lines){
			e.Graphics.DrawLine(Pens.Red,
				line._start.x, 
				line._start.y,
				line._end.x,
				line._end.y);
		}

		for (auto& rect : rects){
			e.Graphics.DrawRectangle(Pens.Red,
				rect._leftUp,
				rect._width,
				rect._height);
		}

		//更改....
		for (auto& circle: circles)
		{
			//.....

		}
		

		
		//...
		Form::OnPaint(e);
	}

};

int main(){
	return 0;
}
