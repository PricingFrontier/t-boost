use t_boost_core::{
    assert_exact_decomposition,
    explain::{check_reconstruction, fixture_model, fixture_serve, RefMeasure},
};

fn main() {
    let model = fixture_model();
    let x = fixture_serve();
    for w in [
        RefMeasure::Uniform,
        RefMeasure::default(),
        RefMeasure::ExposureMarginals { floor: 0.001 },
    ] {
        let bank = model
            .explain_weighted(&x, w.clone(), &[1., 1., 1., 100.])
            .unwrap();
        println!(
            "{w:?}: reconstruction {:?}; assertion {:?}",
            check_reconstruction(&model, &bank),
            assert_exact_decomposition(&model, &bank, &x)
        );
        // control: unit weights
        let bank1 = model.explain_weighted(&x, w.clone(), &[1., 1., 1., 1.]).unwrap();
        println!(
            "   unit-weight control: assertion {:?}; unweighted explain assertion {:?}",
            assert_exact_decomposition(&model, &bank1, &x),
            assert_exact_decomposition(&model, &model.explain(&x, w.clone()).unwrap(), &x)
        );
    }
}
