use crate::Nums;

pub fn sin(el: &[Nums]) -> Vec<f32> {
    el.iter().map(|x| x.sin()).collect::<Vec<Nums>>()
}

pub fn default(el: &[Nums]) -> Vec<f32> {
    el.to_vec()
}