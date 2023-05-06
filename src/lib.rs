
pub type Nums = f32;
pub type FvenResult<T> = Result<T, &'static str>;

pub mod activation;
pub mod loss;
pub mod model;
pub mod prelude;