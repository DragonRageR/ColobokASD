use ASDKyrsova::algorithms::{
    find_candidate_squares, RandomizedSearchSolver, SquareSolver,
};
use ASDKyrsova::geometry::{Point, Square};
use ASDKyrsova::io::FileManager;
use ASDKyrsova::spatial::{KdTree, SpatialHashGrid};

#[test]
fn test_point_parsing_and_formatting() {
    let line = "Point 1: x=22; y=130;";
    let pt = Point::parse(line, 1).expect("Помилка розбору точки");
    assert_eq!(pt.id, 1);
    assert_eq!(pt.x, 22.0);
    assert_eq!(pt.y, 130.0);
    assert_eq!(pt.format_input(), "Point 1: x=22; y=130;");

    // Тест з дробовими координатами
    let line2 = "Point 42: x=12.34; y=56.78;";
    let pt2 = Point::parse(line2, 42).unwrap();
    assert_eq!(pt2.id, 42);
    assert!((pt2.x - 12.34).abs() < 1e-4);
    assert!((pt2.y - 56.78).abs() < 1e-4);
}

#[test]
fn test_rotated_square_detection() {
    // Створюємо повернутий на 45 градусів квадрат:
    // Центр (100, 100), радіус до вершин 50
    // Вершини: (100, 50), (150, 100), (100, 150), (50, 100)
    let points = vec![
        Point::new(1, 100.0, 50.0),
        Point::new(2, 150.0, 100.0),
        Point::new(3, 100.0, 150.0),
        Point::new(4, 50.0, 100.0),
        // Точки-шум
        Point::new(5, 10.0, 10.0),
        Point::new(6, 200.0, 300.0),
    ];

    let candidates = find_candidate_squares(&points, 0.5);
    assert_eq!(candidates.len(), 1, "Повинен бути знайдений 1 повернутий квадрат");

    let solver = RandomizedSearchSolver::new(10);
    let (squares, stats) = solver.solve(&points, 0.5);
    assert_eq!(squares.len(), 1, "Алгоритм повинен знайти повернутий квадрат");
    assert_eq!(stats.disjoint_squares_count, 1);
}

#[test]
fn test_subpixel_tolerance() {
    // Квадрат із субпіксельною похибкою до 0.1 px (в межах допуску 0.5 px)
    let points = vec![
        Point::new(1, 0.0, 0.0),
        Point::new(2, 100.1, 0.1),
        Point::new(3, 100.0, 100.1),
        Point::new(4, -0.1, 100.0),
    ];

    let solver = RandomizedSearchSolver::new(5);
    let (squares, _) = solver.solve(&points, 0.5);
    assert_eq!(squares.len(), 1, "Має знайти квадрат в межах похибки 0.5 px");
}

#[test]
fn test_no_shared_vertices() {
    // Створюємо 2 квадрати, що мають спільну вершину (10, 10),
    // плюс 1 повністю незалежний квадрат.
    let points = vec![
        Point::new(1, 0.0, 0.0),
        Point::new(2, 10.0, 0.0),
        Point::new(3, 10.0, 10.0), // Спільна вершина для квадрата 1 та 2
        Point::new(4, 0.0, 10.0),
        Point::new(5, 20.0, 10.0),
        Point::new(6, 20.0, 20.0),
        Point::new(7, 10.0, 20.0),
        // Квадрат 3 (повністю незалежний)
        Point::new(8, 100.0, 100.0),
        Point::new(9, 120.0, 100.0),
        Point::new(10, 120.0, 120.0),
        Point::new(11, 100.0, 120.0),
    ];

    let solver = RandomizedSearchSolver::new(10);
    let (squares, _) = solver.solve(&points, 0.5);

    // Може бути обрано щонайбільше 2 квадрати
    assert_eq!(squares.len(), 2, "Повинно бути обрано рівно 2 незалежних квадрати");

    // Перевіряємо, що у вибраних квадратах немає жодної спільної вершини
    for i in 0..squares.len() {
        for j in (i + 1)..squares.len() {
            assert!(
                !squares[i].shares_vertex_with(&squares[j]),
                "Квадрати не повинні мати спільних вершин!"
            );
        }
    }
}

#[test]
fn test_spatial_structures_consistency() {
    let points = vec![
        Point::new(1, 10.0, 20.0),
        Point::new(2, 15.0, 25.0),
        Point::new(3, 100.0, 100.0),
        Point::new(4, 102.0, 101.0),
        Point::new(5, 500.0, 500.0),
    ];

    let tree = KdTree::build(&points);
    let grid = SpatialHashGrid::build(&points, 50.0);

    // Шукаємо найближчу точку до (101.0, 100.5)
    let kdtree_res = tree.find_closest_within(&points, 101.0, 100.5, 5.0);
    let grid_res = grid.find_closest_within(&points, 101.0, 100.5, 5.0);

    assert!(kdtree_res.is_some());
    assert!(grid_res.is_some());
    assert_eq!(kdtree_res, grid_res, "KdTree та SpatialHashGrid мають повертати однаковий результат");

    // Пошук у радіусі
    let pts_in_radius_tree = tree.find_within_distance(&points, 12.0, 22.0, 10.0);
    let pts_in_radius_grid = grid.find_within_distance(&points, 12.0, 22.0, 10.0);

    let mut tree_sorted = pts_in_radius_tree;
    tree_sorted.sort();
    let mut grid_sorted = pts_in_radius_grid;
    grid_sorted.sort();

    assert_eq!(tree_sorted, grid_sorted);
    assert_eq!(tree_sorted, vec![0, 1]); // P1 та P2
}

#[test]
fn test_file_io_coursework_format() {
    let sq = Square::new(
        [0, 1, 2, 3],
        [
            Point::new(1, 985.0, 795.0),
            Point::new(2, 960.0, 795.0),
            Point::new(3, 960.0, 45.0),
            Point::new(4, 985.0, 45.0),
        ],
    );

    let out_str = sq.format_output(1);
    assert!(out_str.starts_with("Figure 1: x1=985; y1=795;"));
}

#[test]
fn test_process_sample_input_file() {
    let sample_text = "\
Point 1: x=22; y=130;
Point 2: x=45; y=39;
Point 3: x=985; y=795;
Point 4: x=960; y=795;
Point 5: x=960; y=45;
Point 6: x=985; y=45;
Point 7: x=100; y=100;
Point 8: x=200; y=100;
Point 9: x=200; y=200;
Point 10: x=100; y=200;
Point 11: x=300; y=300;
Point 12: x=350; y=300;
Point 13: x=350; y=350;
Point 14: x=300; y=350;
Point 15: x=400; y=100;
Point 16: x=450; y=150;
Point 17: x=400; y=200;
Point 18: x=350; y=150;
";
    let test_file = "sample_test_io.txt";
    std::fs::write(test_file, sample_text).unwrap();
    let points = FileManager::load_points(test_file).expect("Помилка читання");
    assert_eq!(points.len(), 18);

    let solver = RandomizedSearchSolver::new(15);
    let (squares, stats) = solver.solve(&points, 0.5);
    assert_eq!(squares.len(), 3, "Повинно бути знайдено 3 незалежних квадрати");

    let test_out = "sample_test_out.txt";
    FileManager::save_results(test_out, &squares, &stats).expect("Помилка збереження");

    let out_content = std::fs::read_to_string(test_out).unwrap();
    assert!(out_content.contains("Figure 1: x1="));
    assert!(out_content.contains("Figure 3: x1="));

    let _ = std::fs::remove_file(test_file);
    let _ = std::fs::remove_file(test_out);
}

#[test]
fn test_debug_algorithm_on_2000_points() {
    let points = FileManager::generate_sample_dataset(2000, 1000.0, 800.0, 300);
    assert_eq!(points.len(), 2000);

    let candidates = find_candidate_squares(&points, 0.5);
    assert!(!candidates.is_empty());

    let mut invalid_geom_candidates = 0;
    for c in &candidates {
        if !c.is_valid_square(0.5) {
            invalid_geom_candidates += 1;
        }
    }
    assert_eq!(invalid_geom_candidates, 0, "Усі кандидати мають бути коректними квадратами");

    let solver = RandomizedSearchSolver::new(10);
    let (squares, stats) = solver.solve(&points, 0.5);
    assert!(squares.len() >= 250);

    let mut bad_squares = 0;
    for sq in &squares {
        if !sq.is_valid_square(0.5) {
            bad_squares += 1;
        }
        for (v_idx, &pt_idx) in sq.indices.iter().enumerate() {
            if pt_idx >= points.len() {
                bad_squares += 1;
            } else {
                let p = points[pt_idx];
                let sq_p = sq.points[v_idx];
                if (p.x - sq_p.x).abs() > 1e-4 || (p.y - sq_p.y).abs() > 1e-4 {
                    bad_squares += 1;
                }
            }
        }
    }
    assert_eq!(bad_squares, 0, "Усі обрані фігури повинні відповідати координатам точок");
    assert_eq!(stats.disjoint_squares_count, squares.len());
}

#[test]
fn test_generate_and_verify_with_checkresults() {
    let points = FileManager::generate_sample_dataset(500, 1000.0, 1000.0, 40);
    let temp_data = "temp_test_data.txt";
    let temp_sqs = "temp_test_squares.txt";
    FileManager::save_points(temp_data, &points).unwrap();

    let solver = RandomizedSearchSolver::new(10);
    let (sqs, stats) = solver.solve(&points, 0.5);
    FileManager::save_results(temp_sqs, &sqs, &stats).unwrap();

    for p in &points {
        assert_eq!(p.x.fract(), 0.0);
        assert_eq!(p.y.fract(), 0.0);
    }
    for sq in &sqs {
        assert!(sq.is_valid_square(0.5));
    }

    let _ = std::fs::remove_file(temp_data);
    let _ = std::fs::remove_file(temp_sqs);
}

#[test]
fn test_wide_grid_generation_and_squares_count() {
    // 2000 точок на сітці 1000x800 з кроком 10
    let points = FileManager::generate_sample_dataset(2000, 1000.0, 800.0, 300);
    assert_eq!(points.len(), 2000);

    for p in &points {
        assert!(p.x >= 0.0 && p.x <= 1000.0, "x out of bounds: {}", p.x);
        assert!(p.y >= 0.0 && p.y <= 800.0, "y out of bounds: {}", p.y);
        assert_eq!(p.x % 10.0, 0.0, "x must be multiple of 10: {}", p.x);
        assert_eq!(p.y % 10.0, 0.0, "y must be multiple of 10: {}", p.y);
    }

    let solver = RandomizedSearchSolver::new(10);
    let (sqs, _stats) = solver.solve(&points, 0.5);
    assert!(sqs.len() >= 250, "Очікувалося щонайменше 250 квадратів, знайдено {}", sqs.len());

    // 4000 точок на сітці з кроком 10
    let points_4k = FileManager::generate_sample_dataset(4000, 1000.0, 800.0, 750);
    assert_eq!(points_4k.len(), 4000);
    for p in &points_4k {
        assert_eq!(p.x % 10.0, 0.0, "x must be multiple of 10 on 4k: {}", p.x);
        assert_eq!(p.y % 10.0, 0.0, "y must be multiple of 10 on 4k: {}", p.y);
    }

    let (sqs_4k, stats_4k) = solver.solve(&points_4k, 0.5);
    assert!(sqs_4k.len() >= 600, "Очікувалося щонайменше 600 квадратів для 4k точок, знайдено {}", sqs_4k.len());

    // Зберігаємо результати у файли для зовнішньої валідації CheckResults.exe
    FileManager::save_points("data.txt", &points_4k).unwrap();
    FileManager::save_points("input.txt", &points_4k).unwrap();
    FileManager::save_results("Squares.txt", &sqs_4k, &stats_4k).unwrap();
    FileManager::save_results("Rectangles.txt", &sqs_4k, &stats_4k).unwrap();
    FileManager::save_results("Rombs.txt", &sqs_4k, &stats_4k).unwrap();
    FileManager::save_results("output.txt", &sqs_4k, &stats_4k).unwrap();
}

#[test]
fn test_verify_vulkan_gpu_solver_with_checkresults() {
    use ASDKyrsova::algorithms::VulkanGpuSolver;
    let points = FileManager::generate_sample_dataset(1000, 1000.0, 800.0, 150);
    let solver = VulkanGpuSolver::new(10);
    let (sqs, stats) = solver.solve(&points, 0.5);
    assert!(!sqs.is_empty(), "Vulkan dGPU solver must find squares");

    FileManager::save_points("data.txt", &points).unwrap();
    FileManager::save_points("input.txt", &points).unwrap();
    FileManager::save_results("Squares.txt", &sqs, &stats).unwrap();
    FileManager::save_results("Rectangles.txt", &sqs, &stats).unwrap();
    FileManager::save_results("Rombs.txt", &sqs, &stats).unwrap();
    FileManager::save_results("output.txt", &sqs, &stats).unwrap();
}

