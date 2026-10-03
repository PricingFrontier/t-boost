use t_boost_core::{
    assert_exact_decomposition,
    explain::{
        check_reconstruction, check_three_way_equal, fixture_model, fixture_serve, ExactTol,
        RefMeasure,
    },
};

fn main() {
    let original = fixture_model();
    let x = fixture_serve();
    println!("original explain {:?}", original.explain(&x, RefMeasure::Uniform).map(|_| ()));
    println!("original assert {:?}", assert_exact_decomposition(&original, &original.explain(&x, RefMeasure::Uniform).unwrap(), &x));
    let mut shifted = original.clone();
    shifted.f0 = 10_000_000.0;
    println!("shifted validate {:?}", shifted.validate());
    let mut bank = original.explain(&x, RefMeasure::Uniform).unwrap();
    bank.f0 += 10_000_000.0;
    println!("reconstruct {:?}", check_reconstruction(&shifted, &bank));
    println!("threeway {:?}", check_three_way_equal(&shifted, &bank));
    println!("assert {:?}", assert_exact_decomposition(&shifted, &bank, &x));
    println!("explain {:?}", shifted.explain(&x, RefMeasure::Uniform).map(|_| ()));
    let (mut m1, mut m2, mut centered, mut mass) = (0., 0., 0., 0.);
    for i in 0..3u8 {
        for j in 0..3u8 {
            let v = shifted.ensemble_f64(&[i, j]).unwrap();
            let w = (1. / 3.) * (1. / 3.);
            m1 += w * v;
            m2 += w * v * v;
            centered += w * (v - bank.f0).powi(2);
            mass += w;
        }
    }
    println!(
        "uncentered {}, centered {}, table {}, tolerance {}",
        m2 / mass - (m1 / mass).powi(2),
        centered / mass,
        bank.tables.iter().map(|t| t.variance).sum::<f64>(),
        ExactTol::for_model(&shifted).var_tol
    );
    // scale claim: tree coefficient 1 -> 100000
    let mut scaled = original.clone();
    println!("tree coefficient field: {:?}", scaled.trees[0].0);
    scaled.trees[0].0 = 100000.0;
    println!("scaled validate {:?}", scaled.validate());
    println!("scaled explain {:?}", scaled.explain(&x, RefMeasure::Uniform).map(|_| ()));
    println!("scaled var_tol {}", ExactTol::for_model(&scaled).var_tol);
    for coef in [10.0, 100.0, 1000.0, 10000.0] {
        let mut sc = original.clone();
        sc.trees[0].0 = coef;
        println!("coef {coef}: explain {:?}", sc.explain(&x, RefMeasure::Uniform).map(|_| ()));
    }
    for f0 in [10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0] {
        let mut sh = original.clone();
        sh.f0 = f0;
        println!("f0 {f0}: explain {:?}", sh.explain(&x, RefMeasure::Uniform).map(|_| ()));
    }
}
