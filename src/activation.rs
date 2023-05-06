use std::f32::consts::E;

use crate::Nums;

pub fn default(el: &[Nums]) -> Vec<f32> {
    el.to_vec()
}

pub fn sin(el: &[Nums]) -> Vec<f32> {
    el.iter().map(|x| x.sin()).collect()
}

pub fn relu(el: &[Nums]) -> Vec<f32> {
    el.iter().map(|x| x.max(0.)).collect()
}

pub fn sigmoid(el: &[Nums]) -> Vec<f32> {
    el.iter().map(|x| 1. / (1. + E.powf(*x))).collect()
}
