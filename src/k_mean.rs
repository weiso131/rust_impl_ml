extern crate ndarray;
use ndarray::{Array, ArrayView1, ArrayView2, Ix2, Ix1, s, Axis};

#[derive(Debug)]
#[derive(PartialEq)]
enum Cluster {
    Init,
    ClusterValue(Array<f32, Ix2>),
}

pub struct KMeans {
    cluster_num: usize,
    cluster: Cluster,
}



impl KMeans {
    pub fn new(cluster_num: usize) -> Self {
        Self {
            cluster_num: cluster_num,
            cluster: Cluster::Init
        }
    }
    pub fn train(&mut self, data: &Array<f32, Ix2>) {
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
    pub fn deploy(&mut self, data: &Array<f32, Ix2>) -> Array<usize, Ix1> {
        let mut ans: Array<usize, Ix1> = Array::zeros(data.shape()[0]);
        for i in 0..data.shape()[0] {
            ans[i] = self.get_min_dis_point(&data.slice(s![i, ..]));
        }
        ans
    }

    fn choice_start_point(&self, data: &Array<f32, Ix2>) -> Cluster {
        /*
            生成k_means分類中點的起始點
         */

        if self.cluster_num > data.shape()[0] {
            return Cluster::Init;
        }

        let mut cluster_start_index: Vec<usize> = furthest_centers_heuristic(&self.cluster_num, data);
        let mut cluster_start = Array::zeros((self.cluster_num, data.shape()[1]));
        
        for i in 0..self.cluster_num {
            let cluster_start_point = cluster_start_index[i];
            cluster_start.row_mut(i).assign(&data.slice(s![cluster_start_point, ..]).to_owned());
        }

        Cluster::ClusterValue(cluster_start)

    }

    fn get_min_dis_point(&self, data: &ArrayView1<f32>) -> usize{
        /*
            找到所有分類中點裡面，最接近該筆資料的index
        */


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
        /*
            用來更新分類中點
        */


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

}

fn ndarray_abs(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    let mut dis_total: f32 = 0.0;
    for i in 0..x.shape()[0] {
        dis_total += (x[[i]] - y[[i]]).powi(2);
    }
    dis_total.powf(1.0 / (x.shape()[0] as f32))
}

fn furthest_centers_heuristic(cluster_num: &usize, data: &Array<f32, Ix2>) -> Vec<usize>{
    /*
        Furthest Centers Heuristic Algorithm
        用來確保初始化的中點夠遠
     */


    let mut cluster_start_index: Vec<usize> = Vec::new();//save the index of the data, it will become the cluster start
    cluster_start_index.push(0);

    for _i in 1..*cluster_num {

        let mut max_dis: f32 = 0.0;
        let mut max_index: usize = 0;
        for j in 1..data.shape()[0] {
            let mut dis_total: f32 = 0.0;

            for index in cluster_start_index.iter() {
                dis_total += ndarray_abs(&data.slice(s![*index, ..]), &data.slice(s![j, ..]));
            }

            if dis_total > max_dis {
                max_dis = dis_total;
                max_index = j;
            }

        }
        cluster_start_index.push(max_index);
    }
    cluster_start_index
}