extern crate ndarray;
use ndarray::{Array, ArrayView1, ArrayBase, Ix2, Ix1, s, Axis};

#[derive(Debug)]
#[derive(PartialEq)]
enum Cluster {
    Init,
    ClusterValue(Array<f32, Ix2>),
}

struct KMeans {
    cluster_num: usize,
    cluster: Cluster,
}

fn ndarray_abs(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    let mut dis_total: f32 = 0.0;
    for i in 0..x.shape()[0] {
        dis_total += (x[[i]] - y[[i]]).powi(2);
    }
    dis_total.powf(1.0 / (x.shape()[0] as f32))
}


impl KMeans {
    pub fn new(cluster_num: usize) -> Self {
        Self {
            cluster_num: cluster_num,
            cluster: Cluster::Init
        }
    }
    fn choice_start_point(&self, data: &Array<f32, Ix2>) -> Cluster{
        if self.cluster_num > data.shape()[0] {
            return Cluster::Init;
        }

        let mut cluster_start = Array::zeros((self.cluster_num, data.shape()[1]));

        //will be changed later
        for i in 0..self.cluster_num { 
            cluster_start.row_mut(i).assign(&data.slice(s![i, ..]).to_owned());
        }

        Cluster::ClusterValue(cluster_start)

    }

    fn get_min_dis_point(&self, data: &ArrayView1<f32>) -> usize{
        let mut min_dis: f32 = 1e9;
        let mut index = 0;
        
        if let Cluster::ClusterValue(cluster) = &self.cluster {
            for i in 0..self.cluster_num {
                let dis = ndarray_abs(&cluster.slice(s![i, ..]), &data);
                if dis < min_dis {
                    min_dis = dis;
                    index = i;
                }
            }
        }
        index
    }

    fn get_new_middle_point(&self, data: &Array<f32, Ix2>) -> Array<f32, Ix2> {
        let mut new_middle: Array<f32, Ix2> = Array::zeros((self.cluster_num, data.shape()[1]));
        let mut cluster_class: Array<f32, Ix2> = Array::zeros((self.cluster_num, 1));

        for vector in data.axis_iter(Axis(0)) {
            let class_index = self.get_min_dis_point(&vector);
            
            for i in 0..data.shape()[1] {
                new_middle[[class_index, i]] += vector[i];
            }
            cluster_class[[class_index, 0]] += 1.0
        }
        
        &new_middle / &cluster_class
    }

    pub fn train(&mut self, data: &Array<f32, Ix2>){
        if self.cluster == Cluster::Init { 
            self.cluster = self.choice_start_point(data);
            if self.cluster == Cluster::Init {
                println!("data are smaller than cluster number");
                return ();
            }
        }
        
        for _i in 0..10 {
            let new = self.get_new_middle_point(&data);
            self.cluster = Cluster::ClusterValue(new);
        }
        
    }

}

fn main() {

    let mut model = KMeans::new(2);
    let mut data = Array::<f32, Ix2>::from_shape_vec((30, 2), vec![1.077, 3.842, 1.653, 0.88, 2.85, 2.019, 3.107, 2.039, 1.191, 1.548, 1.034, 1.642, 2.093, 1.969, 1.853, 1.999, 2.736, 1.23, 1.416, 1.895, 1.664, 1.995, 2.645, 1.523, 1.317, 2.344, 1.696, 1.747, 1.519, 0.801, 7.292, 7.618, 8.206, 8.793, 8.623, 8.069, 7.422, 8.069, 7.929, 8.668, 8.385, 8.544, 7.132, 9.257, 7.178, 6.903, 9.222, 8.686, 7.235, 7.424, 9.025, 9.804, 8.663, 7.163, 8.257, 8.395, 8.079, 7.277, 8.571, 7.188]).unwrap();
    
    model.train(&data);

    
}
