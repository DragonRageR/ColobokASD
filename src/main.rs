mod rectangle;
use rectangle::Rectangle;

fn main() {
    let rect1 = Rectangle::new(30, 50);
    rect1.area();
    rect1.print();
    rect1.print_area();
}
