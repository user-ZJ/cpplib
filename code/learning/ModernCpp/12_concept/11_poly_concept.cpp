#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <variant>
using namespace std;


template <typename T>
concept Drawable = requires(T t) {
	t.draw() ;
};

struct Line {
	void draw() const{
		cout<<"Line draw"<<endl;
	}
};

struct Rect{
	void draw() const{
		cout<<"Rect draw"<<endl;
	}
};

template<Drawable T>
void process(const T& shape){
	shape.draw();
}



using Shape = std::variant<Line,Rect>;

int main(){

    Line line;
    Rect rect;

    vector<Shape> vec;
    vec.emplace_back(line);
    vec.emplace_back(rect);
 
    // for_each(vec.begin(), vec.end(), [](Shape& elem){ 
    //         elem.draw();
    //     }
    // );

    for_each(vec.begin(), vec.end(), [](auto& elem){ 
            auto visitor=[](Drawable auto& d){ d.draw();};
            std::visit(visitor,elem);
        }
    );
}
