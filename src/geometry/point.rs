use std::fmt;

/// Точка на двовимірній площині з унікальним ідентифікатором.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub id: usize,
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(id: usize, x: f64, y: f64) -> Self {
        Self { id, x, y }
    }

    /// Евклідова відстань до іншої точки
    pub fn distance_to(&self, other: &Point) -> f64 {
        self.distance_sq(other).sqrt()
    }

    /// Квадрат евклідової відстані
    pub fn distance_sq(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Форматування у форматі вимоги курсової роботи:
    /// "Point 1: x=22; y=130;"
    pub fn format_input(&self) -> String {
        format!("Point {}: x={}; y={};", self.id, self.x.round() as i64, self.y.round() as i64)
    }

    /// Парсинг рядка виду "Point 1: x=22; y=130;" або спрощеного "22 130"
    pub fn parse(line: &str, fallback_id: usize) -> Option<Self> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            return None;
        }

        // Спроба розібрати стандартний формат: Point 1: x=22; y=130;
        if line.contains("x=") && line.contains("y=") {
            let id = if let Some(idx_colon) = line.find(':') {
                let prefix = &line[..idx_colon];
                prefix
                    .split_whitespace()
                    .last()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(fallback_id)
            } else {
                fallback_id
            };

            let x = extract_number_after(line, "x=")?;
            let y = extract_number_after(line, "y=")?;
            return Some(Point::new(id, x, y));
        }

        // Спроба розібрати як два числа через кому або пробіл: "22, 130" або "22 130"
        let parts: Vec<&str> = line
            .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .collect();

        if parts.len() >= 2 {
            if let (Ok(x), Ok(y)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                return Some(Point::new(fallback_id, x, y));
            }
        }

        None
    }
}

fn extract_number_after(text: &str, key: &str) -> Option<f64> {
    let start_idx = text.find(key)? + key.len();
    let rest = &text[start_idx..];
    let end_idx = rest.find(|c: char| c == ';' || c == ',' || c.is_whitespace()).unwrap_or(rest.len());
    rest[..end_idx].trim().parse::<f64>().ok()
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}({:.1}, {:.1})", self.id, self.x, self.y)
    }
}
