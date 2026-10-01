pub struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    pub fn new(width: u32, height: u32) -> Rectangle {
        Rectangle { width, height }
    }

    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    pub fn print(&self) {
        println!("The width of the rectangle is {}", self.width);
        println!("The height of the rectangle is {}", self.height);
    }

    pub fn print_area(&self) {
        println!(
            "The area of the rectangle is {} square pixels.",
            self.area()
        );
    }
}
