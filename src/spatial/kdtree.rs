use crate::geometry::Point;

/// Вузол 2D k-d дерева (деревоподібна структура даних)
pub struct KdNode {
    pub point_idx: usize,
    pub axis: usize, // 0 для X, 1 для Y
    pub left: Option<Box<KdNode>>,
    pub right: Option<Box<KdNode>>,
}

/// 2D k-d дерево для геометричного пошуку за O(log N).
pub struct KdTree {
    pub root: Option<Box<KdNode>>,
    pub count: usize,
}

impl KdTree {
    pub fn new() -> Self {
        Self { root: None, count: 0 }
    }

    /// Побудова k-d дерева зі списку точок за O(N log N)
    pub fn build(points: &[Point]) -> Self {
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let count = indices.len();
        let root = Self::build_recursive(points, &mut indices, 0);
        Self { root, count }
    }

    fn build_recursive(
        points: &[Point],
        indices: &mut [usize],
        depth: usize,
    ) -> Option<Box<KdNode>> {
        if indices.is_empty() {
            return None;
        }

        let axis = depth % 2;

        // Часткове сортування за відповідною віссю (знаходження медіани)
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            let val_a = if axis == 0 { points[a].x } else { points[a].y };
            let val_b = if axis == 0 { points[b].x } else { points[b].y };
            val_a.partial_cmp(&val_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let median_idx = indices[mid];

        let left = Self::build_recursive(points, &mut indices[..mid], depth + 1);
        let right = Self::build_recursive(points, &mut indices[mid + 1..], depth + 1);

        Some(Box::new(KdNode {
            point_idx: median_idx,
            axis,
            left,
            right,
        }))
    }

    /// Пошук усіх точок на відстані <= max_dist від (target_x, target_y)
    pub fn find_within_distance(
        &self,
        points: &[Point],
        target_x: f64,
        target_y: f64,
        max_dist: f64,
    ) -> Vec<usize> {
        let mut result = Vec::new();
        let max_dist_sq = max_dist * max_dist;
        Self::search_recursive(
            &self.root,
            points,
            target_x,
            target_y,
            max_dist,
            max_dist_sq,
            &mut result,
        );
        result
    }

    fn search_recursive(
        node_opt: &Option<Box<KdNode>>,
        points: &[Point],
        tx: f64,
        ty: f64,
        max_dist: f64,
        max_dist_sq: f64,
        result: &mut Vec<usize>,
    ) {
        let Some(node) = node_opt.as_ref() else {
            return;
        };

        let pt = &points[node.point_idx];
        let dx = pt.x - tx;
        let dy = pt.y - ty;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq <= max_dist_sq {
            result.push(node.point_idx);
        }

        let (target_val, node_val) = if node.axis == 0 {
            (tx, pt.x)
        } else {
            (ty, pt.y)
        };

        let delta = target_val - node_val;

        // Обходимо спочатку піддерево, де розташована цільова координата
        let (first, second) = if delta <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        Self::search_recursive(first, points, tx, ty, max_dist, max_dist_sq, result);

        // Якщо гіперсфера перетинає розділову площину, перевіряємо друге піддерево
        if delta.abs() <= max_dist {
            Self::search_recursive(second, points, tx, ty, max_dist, max_dist_sq, result);
        }
    }

    /// Пошук найближчої точки до (target_x, target_y) в межах max_dist без виділення пам'яті
    pub fn find_closest_within(
        &self,
        points: &[Point],
        target_x: f64,
        target_y: f64,
        max_dist: f64,
    ) -> Option<usize> {
        let mut best_dist_sq = max_dist * max_dist;
        let mut best_idx = None;
        Self::closest_recursive(
            &self.root,
            points,
            target_x,
            target_y,
            max_dist,
            &mut best_dist_sq,
            &mut best_idx,
        );
        best_idx
    }

    fn closest_recursive(
        node_opt: &Option<Box<KdNode>>,
        points: &[Point],
        tx: f64,
        ty: f64,
        max_dist: f64,
        best_dist_sq: &mut f64,
        best_idx: &mut Option<usize>,
    ) {
        let Some(node) = node_opt.as_ref() else {
            return;
        };

        let pt = &points[node.point_idx];
        let dx = pt.x - tx;
        let dy = pt.y - ty;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq <= *best_dist_sq {
            *best_dist_sq = dist_sq;
            *best_idx = Some(node.point_idx);
        }

        let (target_val, node_val) = if node.axis == 0 {
            (tx, pt.x)
        } else {
            (ty, pt.y)
        };

        let delta = target_val - node_val;

        let (first, second) = if delta <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        Self::closest_recursive(first, points, tx, ty, max_dist, best_dist_sq, best_idx);

        if delta * delta <= *best_dist_sq {
            Self::closest_recursive(second, points, tx, ty, max_dist, best_dist_sq, best_idx);
        }
    }

    /// Оцінка пам'яті (у байтах)
    pub fn memory_usage(&self) -> usize {
        self.count * std::mem::size_of::<KdNode>() + std::mem::size_of::<Self>()
    }
}
