use fven::prelude::*;


#[test]
fn test() {
    let mut m = Model::new(
        &[
            Layer::new(8, activation::default),
            Layer::new(3, activation::default)
        ],
        1,
        0.005,
        0.0000001
    );

    println!("Before training: {:?}", m.predict(&[5.]));

    for _ in 0..5000 {
        m.train(
            loss::mse, 
            training_data!(
                [1.] => [3., 4., 5.],
                [2.] => [6., 7., 8.],
                [3.] => [9., 10., 11.],
                [4.] => [12., 13., 14.]
            )
        );
        //println!("{:?}", loss::mse(&m.predict(&[5.]), &[15., 16., 17.]));
    }
    
    println!("After training: {:?}", m.predict(&[5.]));

    assert_eq!(&m.predict(&[5.]).iter().map(|v| v.round() as u32).collect::<Vec<u32>>(), &[15, 16, 17])
}
