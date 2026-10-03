use t_boost_core::explain::{fixture_model, fixture_serve, RefMeasure};
use t_boost_core::table_model::TableModel;
fn main() {
    let tm = TableModel::from_model(&fixture_model(), &fixture_serve(), RefMeasure::Uniform).unwrap();
    println!("smoke ok: {} tables", tm.bank.tables.len());
}
