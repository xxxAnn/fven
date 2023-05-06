
use std::rc::Rc;

use rand::Rng;

use crate::FvenResult;
use crate::Nums;

pub struct Model {
    layers: Vec<Layer>,
    __nnums: Vec<(usize, usize)>,
    lr: f32,
    d: f32
}

pub enum Layer {
    Partial(PartialLayer),
    Complete(CompleteLayer)
}
pub struct PartialLayer {
    activation_function: Rc<dyn Fn(&[Nums]) -> Vec<Nums>>,
    number_of_nodes: u32
}

pub struct CompleteLayer {
    act_fun: Rc<dyn Fn(&[Nums]) -> Vec<Nums>>,
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
            activation_function: Rc::new(activation_function)
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
        } else {
            // error 
        }
    }

    pub fn complete(&self, prev: usize) -> Self {
        match self {
            Layer::Partial(p) => {
                let mut rng = rand::thread_rng();

                let nodes = (0..p.number_of_nodes).map(|_| Node {
                    bias: rng.gen(),
                    weights: (0..prev).map(|_| rng.gen()).collect()
                }).collect();

                Layer::Complete(CompleteLayer { act_fun:  p.activation_function.clone(), nodes })
            }
            Layer::Complete(c) => {
                Layer::Complete(CompleteLayer { act_fun: c.act_fun.clone(), nodes: c.nodes.clone() })
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Layer::Complete(c) => {
                c.nodes.len()
            }
            Layer::Partial(p) => {
                p.number_of_nodes as usize
            }
        }
    }

    pub fn calc(&self, inps: &[Nums]) -> FvenResult<Vec<Nums>> {
        if let Layer::Complete(c) = self {
            Ok((c.act_fun)(&c.nodes.iter().map(|n| n.calc(inps).unwrap()).collect::<Vec<Nums>>()))
        } else {
            Err("Attempted to calculate on partially initialized layer")
        }
    }
} 

impl Node {
    pub fn calc(&self, inps: &[Nums]) -> FvenResult<Nums> {
        if inps.len() != self.weights.len() {
            Err("Wrong number of inputs")
        } else {
            Ok((0..inps.len()).map(|i| inps[i]*self.weights[i]).sum::<Nums>() + self.bias)
        }

    }
}

impl Model {

    pub fn new(lyrs: &[Layer], inputs: usize, lr: f32, d: f32) -> Self {
        let mut prev = inputs;
        let mut layers = Vec::new();
        let mut __nnums = Vec::new();
        for l in lyrs.into_iter() {
            let ly = l.complete(prev);
            let clen = ly.len();

            __nnums.push((clen, prev*clen));

            prev = ly.len();
            layers.push(ly);

        }

        Self {
            layers,
            __nnums,
            lr,
            d
        }
    }

    pub fn predict(&self, v: &[Nums]) -> Vec<Nums> {
        let mut o = v.to_vec();
        for layer in &self.layers {
            o = layer.calc(&o).unwrap();
        }
        o
    }

    pub fn get_deltas(&mut self, loss: impl Fn(&[Nums], &[Nums]) -> Nums, data: &[(&[Nums], &[Nums])]) -> Vec<Nums> { // loss(predicted, expected)
        let params = self.get_parameters();
        let mut deltas = Vec::new();
        
        for i in 0..params.len() {
            self.set_parameters(params.clone());
            let fx: Nums = data.iter().map(|dat| loss(&self.predict(dat.0), dat.1)).sum();

            let mut npar = self.get_parameters();
            npar[i] += self.d;
            self.set_parameters(npar);
            let fxpd: Nums = data.iter().map(|dat| loss(&self.predict(dat.0), dat.1)).sum();
            
            deltas.push((fxpd-fx)/self.d)
        }

        self.set_parameters(params);

        deltas
    }

    pub fn train(&mut self, loss: impl Fn(&[Nums], &[Nums]) -> Nums, data: &[(&[Nums], &[Nums])]) {
        let deltas = self.get_deltas(loss, data);
        let mut params = self.get_parameters();

        for i in 0..deltas.len() {
            params[i] -= self.lr*deltas[i];
        }

        self.set_parameters(params)
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