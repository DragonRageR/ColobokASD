use std::time::Instant;
use crate::geometry::{Point, Square};
use super::candidate_detector::find_candidate_squares;
use super::solver::{AlgorithmStats, SquareSolver};

/// Генератор псевдовипадкових чисел (Xorshift64) без додаткових зовнішніх залежностей.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xdeadbeef } else { seed },
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Багатоітераційний рандомізований алгоритм пошуку незалежних квадратів (Randomized Multi-Start).
///
/// Алгоритм розв'язує задачу знаходження максимальної множини взаємно незалежних квадратів (Maximum Independent Set).
/// Складається з двох ключових етапів:
/// 1. Детекція всіх геометричних кандидатів у квадрати за O(n²) за допомогою просторової хеш-сітки.
/// 2. Стохастичний мультистарт: виконання серії проходів жадібного вибору квадратів з мінімальним конфліктом,
///    доповнених випадковим шумом для подолання локальних оптимумів, з вибором найкращого знайденого розв'язку.
pub struct RandomizedSearchSolver {
    pub iterations: usize,
}

use rayon::prelude::*;

impl RandomizedSearchSolver {
    pub fn new(iterations: usize) -> Self {
        Self { iterations }
    }

    /// Стохастичний вибір максимального набору незалежних квадратів на пулі потоків Rayon
    pub fn select_best_randomized(
        candidates: &[Square],
        points_count: usize,
        iterations: usize,
    ) -> Vec<Square> {
        let m = candidates.len();
        if m == 0 {
            return Vec::new();
        }

        // Відображення: індекс точки -> список кандидатів-квадратів, що містять цю точку
        let mut point_to_squares: Vec<Vec<u32>> = vec![Vec::new(); points_count];
        for (sq_idx, sq) in candidates.iter().enumerate() {
            for &pt_idx in &sq.indices {
                if pt_idx < points_count {
                    point_to_squares[pt_idx].push(sq_idx as u32);
                }
            }
        }

        // Багатопотоковий розрахунок базових степенів конфліктності та мінімального покриття точок
        let (base_degrees, min_pt_degrees): (Vec<u32>, Vec<u32>) = (0..m)
            .into_par_iter()
            .map(|sq_idx| {
                let sq = &candidates[sq_idx];
                let mut deg: u32 = 0;
                let mut min_d = u32::MAX;
                for &pt in &sq.indices {
                    if pt < points_count {
                        let cnt = point_to_squares[pt].len() as u32;
                        deg += cnt.saturating_sub(1);
                        min_d = min_d.min(cnt);
                    }
                }
                (deg, if min_d == u32::MAX { 0 } else { min_d })
            })
            .unzip();

        // Паралельний мультистарт: кожна стохастична ітерація виконується незалежно в окремому потоці Rayon
        let num_passes = iterations.max(1);
        let best_solution = (0..num_passes)
            .into_par_iter()
            .map(|pass_idx| {
                let mut rng = SimpleRng::new(123456789 ^ ((pass_idx as u64 + 1) * 2654435761));
                let noise_scale = if pass_idx > 0 {
                    1.0 + (pass_idx % 5) as f64 * 0.5
                } else {
                    0.0
                };

                // Обчислюємо оцінку пріоритету: пріоритет віддається квадратам з «вразливими» точками (min_pt_degrees),
                // що запобігає появі точок-сиріт та радикально знижує похибку відбору
                let mut scores: Vec<(f64, u32)> = (0..m as u32)
                    .map(|idx| {
                        let noise = if pass_idx > 0 {
                            rng.next_f64() * noise_scale
                        } else {
                            0.0
                        };
                        let i = idx as usize;
                        let score = (min_pt_degrees[i] as f64) * 1000.0 + (base_degrees[i] as f64) + noise;
                        (score, idx)
                    })
                    .collect();

                scores.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let mut used_points = vec![false; points_count];
                let mut current_solution = Vec::new();

                for &(_, idx) in &scores {
                    let sq = &candidates[idx as usize];
                    if sq.indices.iter().all(|&p| p < points_count && !used_points[p]) {
                        for &p in &sq.indices {
                            if p < points_count {
                                used_points[p] = true;
                            }
                        }
                        current_solution.push(sq.clone());
                    }
                }

                // Локальна оптимізація 2-Opt / Augment для максимальної незалежної множини (MIS):
                // Шукаємо заміну 1 обраного квадрата на 2 або більше взаємно незалежних кандидатів,
                // що збільшує загальну кількість фігур та рятує точки-сироти.
                for _ in 0..2 {
                    let prev_len = current_solution.len();
                    let mut point_to_sol = vec![usize::MAX; points_count];
                    for (sol_idx, sq) in current_solution.iter().enumerate() {
                        for &p in &sq.indices {
                            if p < points_count {
                                point_to_sol[p] = sol_idx;
                            }
                        }
                    }

                    let mut single_conflict: Vec<Vec<u32>> = vec![Vec::new(); current_solution.len()];
                    for (cand_idx, sq) in candidates.iter().enumerate() {
                        let mut conflict_idx = usize::MAX;
                        let mut conflict_count = 0;
                        for &p in &sq.indices {
                            if p < points_count {
                                let s_idx = point_to_sol[p];
                                if s_idx != usize::MAX && s_idx != conflict_idx {
                                    conflict_idx = s_idx;
                                    conflict_count += 1;
                                    if conflict_count > 1 {
                                        break;
                                    }
                                }
                            }
                        }
                        if conflict_count == 1 {
                            single_conflict[conflict_idx].push(cand_idx as u32);
                        }
                    }

                    let mut augmented_solution = Vec::with_capacity(current_solution.len() + 500);
                    let mut aug_used = vec![false; points_count];

                    for (s_idx, sq) in current_solution.iter().enumerate() {
                        let cand_list = &single_conflict[s_idx];
                        let mut chosen_cands = Vec::new();
                        let mut local_used = vec![false; points_count];

                        for &c_idx in cand_list {
                            let c_sq = &candidates[c_idx as usize];
                            if c_sq.indices.iter().all(|&p| p < points_count && !local_used[p]) {
                                for &p in &c_sq.indices {
                                    if p < points_count {
                                        local_used[p] = true;
                                    }
                                }
                                chosen_cands.push(c_sq.clone());
                                if chosen_cands.len() >= 2 {
                                    break;
                                }
                            }
                        }

                        if chosen_cands.len() >= 2 {
                            // Успішна заміна: 1 квадрат замінюється на 2+ квадрати (+1 до кількості!)
                            for c_sq in chosen_cands {
                                for &p in &c_sq.indices {
                                    if p < points_count {
                                        aug_used[p] = true;
                                    }
                                }
                                augmented_solution.push(c_sq);
                            }
                        } else {
                            for &p in &sq.indices {
                                if p < points_count {
                                    aug_used[p] = true;
                                }
                            }
                            augmented_solution.push(sq.clone());
                        }
                    }

                    // Фінальне жадібне доповнення будь-якими вільними кандидатами
                    for sq in candidates {
                        if sq.indices.iter().all(|&p| p < points_count && !aug_used[p]) {
                            for &p in &sq.indices {
                                if p < points_count {
                                    aug_used[p] = true;
                                }
                            }
                            augmented_solution.push(sq.clone());
                        }
                    }

                    current_solution = augmented_solution;
                    if current_solution.len() <= prev_len {
                        break;
                    }
                }

                current_solution
            })
            .max_by_key(|sol| sol.len())
            .unwrap_or_default();

        best_solution
    }
}

impl SquareSolver for RandomizedSearchSolver {
    fn name(&self) -> &'static str {
        "Рандомізований мультистарт (Randomized Multi-Start)"
    }

    fn description(&self) -> &'static str {
        "Знаходить кандидатів у квадрати через SpatialHashGrid за O(n²) та максимізує кількість незалежних фігур методом багатоітераційного стохастичного пошуку."
    }

    fn complexity(&self) -> &'static str {
        "O(n² + K·m) час, O(n + m) пам'ять"
    }

    fn solve(&self, points: &[Point], tolerance: f64) -> (Vec<Square>, AlgorithmStats) {
        log::info!(
            "Starting square solver on {} points (tolerance: {}, iterations: {})...",
            points.len(),
            tolerance,
            self.iterations
        );

        let start_total = Instant::now();

        // 1. Геометрична детекція кандидатів у квадрати
        let start_detect = Instant::now();
        let candidates = find_candidate_squares(points, tolerance);
        let detection_time_ms = start_detect.elapsed().as_secs_f64() * 1000.0;

        log::debug!(
            "Candidate detection completed in {:.2} ms (found {} candidates)",
            detection_time_ms,
            candidates.len()
        );

        // 2. Стохастичний вибір взаємно незалежної підмножини
        let start_select = Instant::now();
        let chosen = Self::select_best_randomized(&candidates, points.len(), self.iterations);
        let selection_time_ms = start_select.elapsed().as_secs_f64() * 1000.0;

        let total_time_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        log::debug!(
            "Randomized multi-start selection completed in {:.2} ms (selected {} disjoint squares)",
            selection_time_ms,
            chosen.len()
        );

        // Оцінка використання пам'яті
        let mem_points = points.len() * std::mem::size_of::<Point>();
        let mem_cand = candidates.len() * std::mem::size_of::<Square>();
        let mem_graph = candidates.len() * 32;
        let memory_kb = (mem_points + mem_cand + mem_graph) as f64 / 1024.0;

        log::info!(
            "Solver execution completed successfully: {} disjoint squares selected in {:.2} ms (Total memory: {:.1} KB)",
            chosen.len(),
            total_time_ms,
            memory_kb
        );

        let stats = AlgorithmStats {
            algorithm_name: self.name().to_string(),
            complexity: self.complexity().to_string(),
            detection_time_ms,
            selection_time_ms,
            total_time_ms,
            points_count: points.len(),
            candidates_found: candidates.len(),
            disjoint_squares_count: chosen.len(),
            memory_kb,
            hash_variants: self.iterations,
        };

        (chosen, stats)
    }
}
