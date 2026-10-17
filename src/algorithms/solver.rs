use crate::geometry::{Point, Square};

/// Статистика та метрики виконання алгоритму розв'язання.
#[derive(Debug, Clone)]
pub struct AlgorithmStats {
    /// Повна назва використаного алгоритму
    pub algorithm_name: String,
    /// Теоретична обчислювальна складність
    pub complexity: String,
    /// Час детекції геометричних квадратів (у мілісекундах)
    pub detection_time_ms: f64,
    /// Час вибору незалежної підмножини фігур (у мілісекундах)
    pub selection_time_ms: f64,
    /// Загальний час розв'язання задачі (у мілісекундах)
    pub total_time_ms: f64,
    /// Кількість вхідних точок N
    pub points_count: usize,
    /// Кількість знайдених кандидатів у квадрати
    pub candidates_found: usize,
    /// Кількість побудованих взаємно незалежних квадратів
    pub disjoint_squares_count: usize,
    /// Оцінка використаної оперативної пам'яті (у кілобайтах)
    pub memory_kb: f64,
    /// Кількість перевірених хеш варіантів (стохастичний мультистарт)
    pub hash_variants: usize,
}

impl Default for AlgorithmStats {
    fn default() -> Self {
        Self {
            algorithm_name: "Не запускався".to_string(),
            complexity: "-".to_string(),
            detection_time_ms: 0.0,
            selection_time_ms: 0.0,
            total_time_ms: 0.0,
            points_count: 0,
            candidates_found: 0,
            disjoint_squares_count: 0,
            memory_kb: 0.0,
            hash_variants: 40,
        }
    }
}

/// Уніфікований інтерфейс (трайт) для алгоритмів розв'язання задачі курсової роботи №23.
pub trait SquareSolver {
    /// Назва алгоритму для відображення у звіті та інтерфейсі
    fn name(&self) -> &'static str;
    /// Короткий опис принципу роботи
    fn description(&self) -> &'static str;
    /// Теоретична складність за часом та пам'яттю
    fn complexity(&self) -> &'static str;
    /// Запуск повного циклу розв'язання задачі
    fn solve(&self, points: &[Point], tolerance: f64) -> (Vec<Square>, AlgorithmStats);
}
