use rayon::prelude::*;
use crate::geometry::{Point, Square};
use crate::spatial::SpatialHashGrid;

/// Знаходження всіх геометричних кандидатів у квадрати серед множини точок за багатопотоковий час O(n² / P).
///
/// Використовує просторову хеш-таблицю [`SpatialHashGrid`] для швидкої локалізації сусідніх точок за O(1)
/// та пул потоків Rayon для паралельної обробки пар вершин на всіх доступних ядрах CPU.
pub fn find_candidate_squares(points: &[Point], tolerance: f64) -> Vec<Square> {
    let n = points.len();
    if n < 4 {
        return Vec::new();
    }

    // Обчислюємо межі простору та адаптивний розмір комірки для хеш-сітки
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for p in points {
        if p.x < min_x { min_x = p.x; }
        if p.x > max_x { max_x = p.x; }
        if p.y < min_y { min_y = p.y; }
        if p.y > max_y { max_y = p.y; }
    }

    let span = ((max_x - min_x).max(max_y - min_y)).max(10.0);
    let cell_size = (span / (n as f64).sqrt()).max(tolerance * 2.0).max(2.0);

    log::debug!(
        "Starting parallel candidate square detection for {} points (tolerance: {}, cell_size: {:.2})",
        n,
        tolerance,
        cell_size
    );

    let grid = SpatialHashGrid::build(points, cell_size);
    let search_radius = tolerance;

    let max_dist = if n <= 2_000 {
        f64::INFINITY
    } else {
        (cell_size * 10.0).max(140.0)
    };
    let max_dist_sq = max_dist * max_dist;

    // Багатопотоковий паралельний пошук кандидатів у квадрати на Rayon
    let raw_candidates: Vec<Square> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let p_a = points[i];
            let mut local = Vec::new();

            for j in (i + 1)..n {
                let p_b = points[j];
                let dx = p_b.x - p_a.x;
                let dy = p_b.y - p_a.y;
                let dist_sq = dx * dx + dy * dy;

                // Пропускаємо майже однакові або надто віддалені точки
                if dist_sq < 1.0 || dist_sq > max_dist_sq {
                    continue;
                }

                // Вектор повороту на +90 градусів: (-dy, dx)
                // Оскільки i є строго найменшим індексом квадрата (k1 > i && l1 > i && j > i),
                // для кожного опуклого квадрата існує рівно одна реброво-спрямована пара вершин (A -> B),
                // поворот якої на +90 градусів вказує всередину квадрата.
                // Зворотний поворот на -90 градусів є дзеркально надлишковим і створює дублікати.
                let c1_x = p_b.x - dy;
                let c1_y = p_b.y + dx;

                // Швидка перевірка виходу вершини C1 за межі простору (O(1))
                if c1_x < min_x - search_radius || c1_x > max_x + search_radius
                    || c1_y < min_y - search_radius || c1_y > max_y + search_radius
                {
                    continue;
                }

                if let Some(k1) = grid.find_closest_within(points, c1_x, c1_y, search_radius) {
                    if k1 > i && k1 != j {
                        let d1_x = p_a.x - dy;
                        let d1_y = p_a.y + dx;

                        // Швидка перевірка виходу вершини D1 за межі простору (O(1))
                        if d1_x < min_x - search_radius || d1_x > max_x + search_radius
                            || d1_y < min_y - search_radius || d1_y > max_y + search_radius
                        {
                            continue;
                        }

                        if let Some(l1) = grid.find_closest_within(points, d1_x, d1_y, search_radius) {
                            if l1 > i && l1 != j && k1 != l1 {
                                let sq = Square::new(
                                    [i, j, k1, l1],
                                    [p_a, p_b, points[k1], points[l1]],
                                );
                                if sq.is_valid_square(tolerance) {
                                    local.push(sq);
                                }
                            }
                        }
                    }
                }
            }
            local
        })
        .collect();

    // Швидка паралельна канонічна дедуплікація
    let mut keyed_candidates: Vec<([usize; 4], Square)> = raw_candidates
        .into_par_iter()
        .map(|sq| {
            let mut canon = sq.indices;
            canon.sort_unstable();
            (canon, sq)
        })
        .collect();

    keyed_candidates.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
    keyed_candidates.dedup_by(|a, b| a.0 == b.0);

    // Сортуємо кандидатів за зростанням площі (найменші квадрати мають найвищий пріоритет для пакування)
    keyed_candidates.par_sort_unstable_by_key(|(_, sq)| {
        let s = sq.side_length();
        (s * s) as u64
    });

    if keyed_candidates.len() > 800_000 {
        keyed_candidates.truncate(800_000);
    }

    let candidates: Vec<Square> = keyed_candidates
        .into_iter()
        .map(|(_, sq)| sq)
        .collect();

    log::debug!(
        "Candidate square detection finished: found {} valid candidate squares",
        candidates.len()
    );

    candidates
}
