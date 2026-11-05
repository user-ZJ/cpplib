class MainForm : public Form {
private:
	Point p1;
	Point p2;

	//vector<Shape> shapes;
	//vector<Shape&> shapes;
	//vector<Shape*> shapes;
	std::vector<std::unique_ptr<Shape>> shapes;
	
public:
	MainForm(){
		
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
		
		//...

		//创建对象
		// ...make_unique<Line>();
		// ...make_unique<Rect>();


		//...
		this->Refresh();

		Form::OnMouseUp(e);
	}

	void MainForm::OnPaint(const PaintEventArgs& e) override {

		for (auto& shape : shapes){
			shape->Draw(e.Graphics); //多态调用
		}

		Form::OnPaint(e);
	}

	
};
