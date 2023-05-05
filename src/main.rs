pub type Nums = f32;

pub struct Model {
    layers: Vec<Layer>,
    __nnums: Vec<(usize, usize)>
}

pub enum Layer {
    Partial(PartialLayer),
    Complete(CompleteLayer)
}

pub struct PartialLayer {
    activation_function: Box<dyn Fn(&[Nums]) -> Vec<Nums>>,
    number_of_nodes: u32
}

pub struct CompleteLayer {
    act_fun: Box<dyn Fn(&[Nums]) -> Vec<Nums>>,
    nodes: Vec<Node>
}

#[derive(Clone)]
pub struct Node {
    bias: Nums,
    weights: Vec<Nums>
}

impl Layer {
    pub fn new(number_of_nodes: u32, activation_function: impl Fn(&[Nums]) -> Vec<Nums> + 'static) -> Self {
        Self::Partial(PartialLayer {
            number_of_nodes,
            activation_function: Box::new(activation_function)
        })
    }

    pub fn update(&mut self, biases: &[f32], weights: &[f32]) {
        if let Layer::Complete(c) = self {
            let mut i = 0;
            let mut u = 0;
            let wlen = weights.len()/biases.len();
            for node in &mut c.nodes {
                node.bias = biases[i];
                node.weights = weights[u..u+wlen].to_vec();
                u += wlen;
                i += 1
            }
        }
    }
}

impl Model {

    pub fn new(lyrs: &[Layer]) -> Self {
        todo!()
    }

    pub fn get_parameters(&self) -> Vec<Nums> {
        self.layers.iter().map(|l| {
            if let Layer::Complete(ly) = l {
                vec![
                    ly
                        .nodes
                        .clone()
                        .into_iter()
                        .map(|n| n.bias)
                        .collect::<Vec<Nums>>(),

                    ly
                        .nodes
                        .clone()
                        .into_iter()
                        .map(|n| n.weights)
                        .into_iter()
                        .flatten().collect::<Vec<Nums>>()
                ]
                    .into_iter()
                    .flatten()
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect::<Vec<Vec<Nums>>>()
        .into_iter()
        .flatten()
        .collect()
    }

    pub fn set_parameters(&mut self, parameters: Vec<Nums>) {
        if self.__nnums.len() != self.layers.len() {
            // error
        }
        let mut i = 0;
        let mut u = 0;
        for layer in &mut self.layers {
            let plus = self.__nnums[i];
            let biases = &parameters[u..u+plus.0];
            u += plus.0;
            let weights = &parameters[u..u+plus.1];
            u += plus.1;
            
            layer.update(biases, weights);
            
            i += 1
        }
    }
}


fn sin(el: &[Nums]) -> Vec<f32> {
    el.iter().map(|x| x.sin()).collect::<Vec<Nums>>()
}
fn main() {
    Layer::new(4, sin);
}
