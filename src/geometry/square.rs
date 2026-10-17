use super::point::Point;

/// Квадрат, утворений 4 точками у циклічному порядку (A -> B -> C -> D).
#[derive(Debug, Clone, PartialEq)]
pub struct Square {
    /// Індекси вершин у загальному списку точок
    pub indices: [usize; 4],
    /// Координати 4 вершин у порядку обходу
    pub points: [Point; 4],
}

impl Square {
    pub fn new(indices: [usize; 4], points: [Point; 4]) -> Self {
        Self { indices, points }
    }

    /// Повертає відсортований масив індексів для унікальної ідентифікації набору вершин
    pub fn canonical_indices(&self) -> [usize; 4] {
        let mut sorted = self.indices;
        sorted.sort_unstable();
        sorted
    }

    /// Перевіряє, чи мають два квадрати спільні вершини
    pub fn shares_vertex_with(&self, other: &Square) -> bool {
        for &i1 in &self.indices {
            for &i2 in &other.indices {
                if i1 == i2 {
                    return true;
                }
            }
        }
        false
    }

    /// Довжина сторони квадрата
    pub fn side_length(&self) -> f64 {
        self.points[0].distance_to(&self.points[1])
    }

    /// Площа квадрата
    pub fn area(&self) -> f64 {
        let s = self.side_length();
        s * s
    }

    /// Периметр квадрата
    pub fn perimeter(&self) -> f64 {
        self.side_length() * 4.0
    }

    /// Центр квадрата
    pub fn center(&self) -> (f64, f64) {
        let sum_x: f64 = self.points.iter().map(|p| p.x).sum();
        let sum_y: f64 = self.points.iter().map(|p| p.y).sum();
        (sum_x / 4.0, sum_y / 4.0)
    }

    /// Форматування згідно з вимогами курсової роботи (п. 4.2):
    /// "Figure 1: x1=985; y1=795; x2=960; y2=795; x3=960; y3=45; x4=985; y4=45;"
    pub fn format_output(&self, figure_id: usize) -> String {
        format!(
            "Figure {}: x1={}; y1={}; x2={}; y2={}; x3={}; y3={}; x4={}; y4={};",
            figure_id,
            self.points[0].x.round() as i64,
            self.points[0].y.round() as i64,
            self.points[1].x.round() as i64,
            self.points[1].y.round() as i64,
            self.points[2].x.round() as i64,
            self.points[2].y.round() as i64,
            self.points[3].x.round() as i64,
            self.points[3].y.round() as i64,
        )
    }

    /// Перевірка чи фігура є квадратом з точністю tolerance (згідно з валідатором CheckResults.exe)
    pub fn is_valid_square(&self, tolerance: f64) -> bool {
        let tol = tolerance.max(1e-5);

        let s0 = self.points[0].distance_to(&self.points[1]);
        let s1 = self.points[1].distance_to(&self.points[2]);
        let s2 = self.points[2].distance_to(&self.points[3]);
        let s3 = self.points[3].distance_to(&self.points[0]);

        if s0 < 1.0 || s1 < 1.0 || s2 < 1.0 || s3 < 1.0 {
            return false;
        }

        // 1. Усі 4 сторони мають бути рівними між собою в межах tolerance
        // (згідно з валідатором CheckResults.exe: max_side - min_side <= tol)
        let min_s = s0.min(s1).min(s2).min(s3);
        let max_s = s0.max(s1).max(s2).max(s3);

        if (max_s - min_s) > tol {
            return false;
        }

        // 2. Довжини обох діагоналей мають бути рівними в межах tolerance
        // (згідно з валідатором CheckResults.exe: |diag1 - diag2| <= tol)
        let diag1 = self.points[0].distance_to(&self.points[2]);
        let diag2 = self.points[1].distance_to(&self.points[3]);

        if (diag1 - diag2).abs() > tol {
            return false;
        }

        if diag1 < 1.0 || diag2 < 1.0 {
            return false;
        }

        true
    }
}
