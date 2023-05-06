use crate::Nums;

pub fn mse(el: &[Nums], elp: &[Nums]) -> Nums {
    (0..el.len()).map(|i| (el[i]-elp[i]).powi(2)).sum::<Nums>()/el.len() as Nums
}