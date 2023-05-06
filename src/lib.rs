
pub type Nums = f32;
pub type FvenResult<T> = Result<T, &'static str>;

pub mod activation;
pub mod loss;
pub mod model;
pub mod prelude;


#[macro_export]
macro_rules! training_data {
    ($($i:expr => $o:expr),+) => (
        &[
            $((
                &$i,
                &$o
            )),+
        ]
    )
}