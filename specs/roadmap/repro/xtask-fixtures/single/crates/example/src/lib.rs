#[derive(Serialize, Deserialize)]
pub struct Bad {
    pub n: usize,
    pub m: HashMap<u32, u32>,
}
