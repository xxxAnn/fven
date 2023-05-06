use fven::prelude::*;

#[test]
fn test() {
    let mut m = Model::new(
        &[
            Layer::new(2, activation::default),
            Layer::new(3, activation::default)
        ],
        1
    );

    println!("Before training: {:?}", m.predict(&[3.]));

    for _ in 0..10000 {
        m.train(loss::mse, 
            &[
                (
                    &[1.], // inputs
                    &[3., 4., 5.] // outputs
                ),
                (
                    &[2.],
                    &[6., 7., 8.]
                ),
                (
                    &[3.],
                    &[9., 10., 11.]
                ),
                (
                    &[4.],
                    &[12., 13., 14.]
                )
            ]
        );
    }
    
    assert_eq!(&m.predict(&[5.]).iter().map(|v| v.round() as u32).collect::<Vec<u32>>(), &[15, 16, 17])
}
