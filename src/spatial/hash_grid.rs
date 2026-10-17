use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use crate::geometry::Point;

/// Швидкий некриптографічний хешер для просторових індексів
#[derive(Default)]
pub struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ (b as u64)).wrapping_mul(0x517cc1b727220a95);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        let mut x = i;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        self.0 = x;
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.0 = (self.0 ^ (i as u64)).wrapping_mul(0x517cc1b727220a95);
    }
}

pub type FastBuildHasher = BuildHasherDefault<FastHasher>;

/// Просторова хеш-сітка (Spatial Hash Grid) на основі швидкої хеш-таблиці.
/// Забезпечує пошук точок у радіусі r за середній час O(1).
pub struct SpatialHashGrid {
    cell_size: f64,
    grid: HashMap<(i32, i32), Vec<usize>, FastBuildHasher>,
}

impl SpatialHashGrid {
    pub fn new(cell_size: f64) -> Self {
        Self {
            cell_size: if cell_size <= 0.0 { 1.0 } else { cell_size },
            grid: HashMap::with_hasher(FastBuildHasher::default()),
        }
    }

    /// Створює хеш-сітку з заданого списку точок
    pub fn build(points: &[Point], cell_size: f64) -> Self {
        let mut grid = Self::new(cell_size);
        for (idx, pt) in points.iter().enumerate() {
            grid.insert(idx, pt);
        }
        grid
    }

    /// Обчислює координати комірки для заданих координат (x, y)
    #[inline]
    fn cell_coords(&self, x: f64, y: f64) -> (i32, i32) {
        let cx = (x / self.cell_size).floor() as i32;
        let cy = (y / self.cell_size).floor() as i32;
        (cx, cy)
    }

    /// Додавання точки до сітки
    pub fn insert(&mut self, point_index: usize, point: &Point) {
        let cell = self.cell_coords(point.x, point.y);
        self.grid.entry(cell).or_default().push(point_index);
    }

    /// Пошук усіх точок, відстань від яких до (target_x, target_y) <= max_dist
    pub fn find_within_distance(
        &self,
        points: &[Point],
        target_x: f64,
        target_y: f64,
        max_dist: f64,
    ) -> Vec<usize> {
        let mut result = Vec::new();
        let max_dist_sq = max_dist * max_dist;

        let min_cell = self.cell_coords(target_x - max_dist, target_y - max_dist);
        let max_cell = self.cell_coords(target_x + max_dist, target_y + max_dist);

        for cx in min_cell.0..=max_cell.0 {
            for cy in min_cell.1..=max_cell.1 {
                if let Some(indices) = self.grid.get(&(cx, cy)) {
                    for &idx in indices {
                        let pt = &points[idx];
                        let dx = pt.x - target_x;
                        let dy = pt.y - target_y;
                        if dx * dx + dy * dy <= max_dist_sq {
                            result.push(idx);
                        }
                    }
                }
            }
        }

        result
    }

    /// Пошук найближчої точки до (target_x, target_y) в межах max_dist без виділення пам'яті (O(1))
    #[inline]
    pub fn find_closest_within(
        &self,
        points: &[Point],
        target_x: f64,
        target_y: f64,
        max_dist: f64,
    ) -> Option<usize> {
        let max_dist_sq = max_dist * max_dist;
        let min_cell = self.cell_coords(target_x - max_dist, target_y - max_dist);
        let max_cell = self.cell_coords(target_x + max_dist, target_y + max_dist);

        let mut closest_idx = None;
        let mut min_d_sq = max_dist_sq;

        // Швидкий шлях: коли радіус пошуку повністю вкладається в одну комірку
        if min_cell == max_cell {
            if let Some(indices) = self.grid.get(&min_cell) {
                for &idx in indices {
                    let pt = &points[idx];
                    let dx = pt.x - target_x;
                    let dy = pt.y - target_y;
                    let d_sq = dx * dx + dy * dy;
                    if d_sq <= min_d_sq {
                        min_d_sq = d_sq;
                        closest_idx = Some(idx);
                    }
                }
            }
            return closest_idx;
        }

        for cx in min_cell.0..=max_cell.0 {
            for cy in min_cell.1..=max_cell.1 {
                if let Some(indices) = self.grid.get(&(cx, cy)) {
                    for &idx in indices {
                        let pt = &points[idx];
                        let dx = pt.x - target_x;
                        let dy = pt.y - target_y;
                        let d_sq = dx * dx + dy * dy;
                        if d_sq <= min_d_sq {
                            min_d_sq = d_sq;
                            closest_idx = Some(idx);
                        }
                    }
                }
            }
        }

        closest_idx
    }

    /// Оцінка використаної пам'яті структурою (у байтах)
    pub fn memory_usage(&self) -> usize {
        let entry_size = std::mem::size_of::<(i32, i32)>() + std::mem::size_of::<Vec<usize>>();
        let elements_size: usize = self.grid.values().map(|v| v.capacity() * std::mem::size_of::<usize>()).sum();
        std::mem::size_of::<Self>() + (self.grid.capacity() * entry_size) + elements_size
    }
}
