use std::process::exit;
use std::fs::File;
use std::io::{self,BufRead};
use std::path::Path;
use std::env;

use std::collections::HashMap;
use std::str::FromStr;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::Entry;
use std::collections::VecDeque;
//use priority_queue::DoublePriorityQueue;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use std::result::Result;

#[derive(Debug)]
#[derive(Clone)]
#[derive(Copy)]
struct Vertex{
    id: u32,
    x: f64,
    y: f64
}
impl Hash for Vertex{
    fn hash<H:Hasher>(&self, state: &mut H){
        self.id.hash(state);
    }
}
impl PartialEq for Vertex{
    fn eq(&self , other: &Self  ) -> bool{
        self.id == other.id
    }
}
impl Eq for Vertex{}
impl Vertex{
    fn get_x(&self)-> &f64{
        &self.x
    }
    fn get_y(&self)-> &f64{
        &self.y
    }
    fn get_id(&self)-> &u32{
        &self.id
    }
    //fn calculate_edge_average(&self, adyacencies_list)
}
#[derive(Debug)]
#[derive(Clone)]
#[derive(Copy)]
#[derive(PartialEq, Eq, Hash)]
struct Edge{
    vi : Vertex,
    vf : Vertex
}
/*
impl PartialEq for Edge{
    fn eq(&self , other: &Self  ) -> bool{
       (self.vi.eq( (*other).get_vi() ) &&  self.vf.eq( (*other).get_vf() ) )
       || (self.vi.eq( (*other).get_vf() ) &&  self.vf.eq( (*other).get_vi() ) )
    }
}
impl Eq for Edge {}
*/
impl Edge{
    fn get_weigth(&self) -> f64{
        let xf_minus_xi = self.vf.get_x() - self.vi.get_x();
        let yf_minus_yi = self.vf.get_y() - self.vi.get_y();

        (xf_minus_xi*xf_minus_xi + yf_minus_yi*yf_minus_yi).sqrt()
    }
    fn get_vi(&self) -> &Vertex{
        &self.vi
    }
    fn get_vf(&self) -> &Vertex{
        &self.vf
    }
    fn vertex_in_edge(&self, v: Vertex)-> bool{
        if self.vi.eq(&v){
            return true ;
        }
        if self.vf.eq(&v){
            return true ;
        }
        false
    }
}
#[derive(Debug)]
#[derive(Clone)]
#[derive(Copy)]
enum Adjacency{
    Empty,
    NonEmpty(Edge)
}
#[derive(Debug)]
#[derive(Clone)]
struct Graph{
    size : usize,
    vertexs : HashMap<u32,Vertex>,
    adjacencies_matrix  : Vec<Adjacency>
}
impl Graph{

    fn new(vertexs_list : Vec<Vertex>) -> Graph{
        let size =  vertexs_list.len();
        let v_size =  vertexs_list.len()+1;
        let mut adjacent_vector : Vec<Adjacency> = vec![Adjacency::Empty; v_size*v_size ];

        /*
         * Building G this way we ensure we have G[i][j] = edge=(vi=i,vf=j)
         */
        for i in 0..size-1{
            let vi : &Vertex = &vertexs_list[i];
            for j in (i+1)..size{
                let vf : &Vertex = &vertexs_list[j];

                let id_vi : usize = *vi.get_id() as usize;
                let id_vf : usize = *vf.get_id() as usize;

                let index_a : usize = (id_vi*v_size + id_vf) as usize;
                let index_b : usize = (id_vf*v_size + id_vi) as usize;

                let e : Edge = Edge{ vi:*vi, vf: *vf };

                adjacent_vector[index_a] = Adjacency::NonEmpty( e );
                adjacent_vector[index_b] = Adjacency::NonEmpty( e );
            }
        }

        let mut vertex_dictionary : HashMap<u32, Vertex> = HashMap::new();
        for v in vertexs_list{
            vertex_dictionary.insert(
                v.id,
                v
            );
        }

        Graph{
            size: v_size,
            vertexs: vertex_dictionary,
            adjacencies_matrix: adjacent_vector
        }
    }

    fn get_edges_adjacent_to_vertex(&mut self, v: Vertex)-> Vec<Edge>{
        //edges (v,u) are in a single row, except by identity.
        let index = *v.get_id() as usize;
        let size = self.get_size();
        let mut adjacent_edges: Vec<Edge> = Vec::new();
        let mut c = 1;
        for i in (index*size)..((index+1)*size){
            c += 1;
            match self.adjacencies_matrix[i]{
                Adjacency::NonEmpty(e) => adjacent_edges.push(e),
                _ => {}
            };
        }
        adjacent_edges
    }

    fn get_edges_adyacent_to_tree(&mut self, tree_vertexs: HashMap<u32,Vertex>, tree_edges:Vec<Edge>)-> Vec<Edge>{


        let mut edges_adyacent: Vec<Edge> = Vec::new();
        for (_,v) in tree_vertexs{
            let edges_adyacent_to_v = self.get_edges_adjacent_to_vertex(v);
            for e in edges_adyacent_to_v{
                if ! tree_edges.contains(&e){
                    edges_adyacent.push(e);
                }
            }
        }

        edges_adyacent
    }

    fn get_edges_from_vertex_to_tree(&mut self, v: Vertex, tree: &mut Tree) -> Vec<Edge>{
        let mut edges : Vec<Edge> = Vec::new();
        let v_id = *v.get_id() as usize;

        for (w_id , _ ) in tree.get_tree_vertexs(){
            match self.get_adjacency( (w_id as usize), v_id){
                Adjacency::NonEmpty(e) => edges.push(e),
                _ => {}
            }
        }

        edges

    }

    fn get_edges_not_in_list( &mut self, edge_list: &Vec<Edge>)->Vec<Edge>{

        let mut list : Vec<Edge> = Vec::new();

        for i in 0..self.size{
            for j in (i+1)..self.size{
                let a: Adjacency = self.get_adjacency(i, j);
                match a{
                    Adjacency::NonEmpty(e)=>{
                        if ! edge_list.contains(&e) {
                            list.push(e);
                        }
                    },
                    _ =>{}
                }
            }
        }

        list
    }

    fn get_adjacency(&mut self, i: usize, j: usize)->Adjacency{
        self.adjacencies_matrix[i*self.size + j]
    }

    fn get_size(&self)-> usize{
        self.size.clone()
    }

    fn get_vertex(&self, id_vertex: u32)->Vertex{
        self.vertexs[&id_vertex]
    }

    fn get_vertex_dictionary(&self)->&HashMap<u32,Vertex>{
        &self.vertexs
    }

}


#[derive(Debug)]
#[derive(Clone)]
struct Tree{
    tree_vertex_dictionary : HashMap<u32, Vertex>,
    tree_edge_list : Vec<Edge>
}
impl Tree{
    fn new() -> Tree{
        let vertex_dictionary : HashMap<u32,Vertex> = HashMap::new();
        let edge_list : Vec<Edge> = Vec::new();
        Tree{
            tree_vertex_dictionary: vertex_dictionary,
            tree_edge_list: edge_list
        }
    }


    fn add_edge_to_tree(&mut self, e: Edge)->bool{

        //Add edge if not in list
        if self.tree_edge_list.contains(&e){
            return false;
        }

        let vi : &Vertex = e.get_vi();
        let vf : &Vertex = e.get_vf();

        let contains_vi = self.tree_vertex_dictionary.contains_key(
            vi.get_id()
        );
        let contains_vf = self.tree_vertex_dictionary.contains_key(
            vf.get_id()
        );

        //Add vertex if not in list
        if ! contains_vi {
            self.tree_vertex_dictionary.insert(*vi.get_id(), *vi);
        }
        if ! contains_vf {
            self.tree_vertex_dictionary.insert(*vf.get_id(), *vf);
        }

        self.tree_edge_list.push(e);

        true

    }

    fn build_neighbour_tree(&mut self)-> HashMap<u32, Vec<Vertex>>{
        //First we build our neighbour tree from vertex
        let mut vertex_neighbours : HashMap<u32, Vec<Vertex> > = HashMap::new();

        for e in &self.tree_edge_list{

            let v_i = e.get_vi();
            let v_f = e.get_vf();

            match vertex_neighbours.entry(*v_i.get_id()) {
                Entry::Vacant(entry) =>{ entry.insert(vec![*v_f]); },
                Entry::Occupied(mut entry)=>{ entry.get_mut().push(*v_f);}
            };

            match vertex_neighbours.entry(*v_f.get_id()) {
                Entry::Vacant(entry) =>{ entry.insert(vec![*v_i]); },
                Entry::Occupied(mut entry)=>{ entry.get_mut().push(*v_i);}
            };


        }
        
        vertex_neighbours
    }

    fn get_u_v_unique_path( &mut self, u: Vertex, v: Vertex )->Vec<Edge>{

        let v_id = v.get_id();
        let u_id = u.get_id();

        let vertex_neighbours : HashMap<u32, Vec<Vertex> > = self.build_neighbour_tree();
        //Find unique trayectory. BFS because is nicer for tree
        let mut queue_bfs: VecDeque<Vertex> = VecDeque::new();
        let mut visited_register : HashMap<u32, bool> = HashMap::new();
        let mut parents: HashMap<u32, u32> = HashMap::new();
        let mut parents_v: HashMap<u32, Vertex> = HashMap::new();

        //Anyone visited nor has parent
        for (k, _) in &self.tree_vertex_dictionary{
           visited_register.insert(*k, false);
            //parents.insert(*k, 0);
        }

        *visited_register.get_mut(v_id).unwrap() = true;
        //*parents.get_mut(v_id).unwrap() = *v_id;
        parents.insert(*v_id, *v_id);
        queue_bfs.push_front(v);

        let mut tray_found: bool = false;

        while queue_bfs.len() != 0 && !tray_found{
            let w : Vertex = queue_bfs.pop_front().unwrap();

            tray_found = w.eq(&u);

            for r  in vertex_neighbours[w.get_id()].clone(){

                if ! visited_register.get(r.get_id()).unwrap() {

                    *visited_register.get_mut(r.get_id()).unwrap() = true;
                    parents.insert(*r.get_id(), *w.get_id() );
                    //*parents_v.get_mut(r.get_id()).unwrap() = w;
                    parents_v.insert(*r.get_id(), w);
                    //parents[r.get_id()] = w;
                    queue_bfs.push_back(r);
                }
            }
        }
        //Build tray


        let mut tray: Vec<Edge> = Vec::new();

        if tray_found {

            let mut first : Vertex = u;
            let mut next: Vertex = *parents_v.get(first.get_id()).unwrap();

            tray.push(Edge{vi: first, vf: next});

            while( !v.eq( &next) ){
                first = next;
                next = *parents_v.get(first.get_id()).unwrap();
                let e = Edge{vi: first, vf: next};
                tray.push(e);
            }
        }

        tray
    }

    fn get_tree_weigth(&self)-> f64{
        let mut weigh = 0.0;
        for e in &self.tree_edge_list{
            weigh += e.get_weigth();
        }
        weigh
    }

    fn get_connected_component_deleting_vertex(&mut self, v: Vertex)-> Vec<Tree>{
        //Build neighbour tree:
        let vertex_neighbours : HashMap<u32, Vec<Vertex> > = self.build_neighbour_tree();
        let mut visited_register : HashMap<u32, bool> = HashMap::new();

        for (k, _) in &self.tree_vertex_dictionary{
           visited_register.insert(*k, false);
        }

        let vertex_sub_trees: Vec<Vertex> = vertex_neighbours.get(v.get_id()).unwrap().clone();
        *visited_register.get_mut(v.get_id()).unwrap() = true;

        let mut tree_components : Vec<Tree> = Vec::new();

        for root in vertex_sub_trees{
            //Actual BFS
            //let root_tree = vertex_neighbours.get(root.get_id());

            let mut queue_bfs: VecDeque<Vertex> = VecDeque::new();
            let mut edges_of_tree : Vec<Edge> = Vec::new();

            *visited_register.get_mut(root.get_id()).unwrap() = true;
            queue_bfs.push_front(root);
            while queue_bfs.len() != 0 {
                let w : Vertex = queue_bfs.pop_front().unwrap();

                for ww in vertex_neighbours[w.get_id()].clone(){
                    if ! visited_register.get(ww.get_id()).unwrap(){

                        *visited_register.get_mut(ww.get_id()).unwrap() = true;
                        let new_edge = Edge{vi: w, vf:ww};
                        edges_of_tree.push( new_edge );

                        queue_bfs.push_back(ww);
                    }
                }

            }

            let mut component : Tree = Tree::new();
            for e in edges_of_tree{
                component.add_edge_to_tree(e);
            }

            tree_components.push(component);

        }

        tree_components
    }

    //To keep uncycled
    fn add_edge_to_tree_keeping_acyclic(&mut self, e: Edge)-> bool{
        //Add edge if not in list
        if self.tree_edge_list.contains(&e){
            return false;
        }

        let vi : &Vertex = e.get_vi();
        let vf : &Vertex = e.get_vf();

        let contains_vi = self.tree_vertex_dictionary.contains_key(
            vi.get_id()
        );
        let contains_vf = self.tree_vertex_dictionary.contains_key(
            vf.get_id()
        );

        //Not add cycles
        if contains_vi && contains_vf{
            return false;
        }

        //Add vertex if not in list
        if ! contains_vi {
            self.tree_vertex_dictionary.insert(*vi.get_id(), *vi);
        }
        if ! contains_vf {
            self.tree_vertex_dictionary.insert(*vf.get_id(), *vf);
        }

        self.tree_edge_list.push(e);

        true
    }
    fn delete_edge(&mut self, e: Edge ){

        if self.tree_edge_list.contains(&e){
            let index =  self.tree_edge_list.iter().position(|x| x.eq(&e) ).unwrap();
            self.tree_edge_list.remove(index);

            let v_i = e.get_vi();
            let v_f = e.get_vf();

            let mut v_i_still_adyacent = false;
            let mut v_f_still_adyacent = false;

            for ee in &self.tree_edge_list{
                if ee.vertex_in_edge(*v_i){
                    v_i_still_adyacent = true;
                }
                if ee.vertex_in_edge(*v_f){
                    v_f_still_adyacent = true;
                }
            }

            if ! v_i_still_adyacent{
                self.tree_vertex_dictionary.remove(v_i.get_id());
            }
            if ! v_f_still_adyacent{
                self.tree_vertex_dictionary.remove(v_f.get_id());
            }
        }
    }

    fn get_tree_vertexs(&mut self)->HashMap<u32, Vertex>{
        self.tree_vertex_dictionary.clone()
    }
    fn get_tree_edges(&mut self)->Vec<Edge>{
        self.tree_edge_list.clone()
    }
    /*
    fn find_edge_by_vertices(&mut self, v: Vertex, w: Vertex)->Edge{
        for
    }
    */
}

struct MinimumKTreeHeuristic{
    graph: Graph,
    random_generator: StdRng,
    v_out: HashMap<u32, Vertex>,
    v_in: HashMap<u32, Vertex>,
    tt_inc: u32,
    tt_min: u32,
    tt_max: u32,
    tt_ten: u32,
    nic_int : u32,
    nic_max: u32,
    max_num_iter: u32
}

impl MinimumKTreeHeuristic{

    fn new(g: Graph, rand: StdRng, vertex_number: u32, k: u32) -> MinimumKTreeHeuristic{

        let mut tt_min_options : Vec<u32>= vec![vertex_number/20,
                                                (vertex_number-k)/4,
                                                k/4
                                                ];
        tt_min_options.sort_by(|a,b| b.cmp(a) );
        let tt_min = tt_min_options.pop().unwrap();
        let tt_max = vertex_number/5;

        let tt_inc = ((tt_max-tt_min)/10) +1;
        let nic_int = 0;
        let nic_max = if tt_inc > 100 {
                            tt_inc.clone()
                       }else{
                            100
                       };

        MinimumKTreeHeuristic{
            graph: g,
            random_generator: rand,
            v_out : HashMap::new(),
            v_in : HashMap::new(),
            tt_inc: tt_inc,
            tt_min: tt_min,
            tt_max: tt_max,
            tt_ten: tt_min,
            nic_int: nic_int,
            nic_max: nic_max,
            max_num_iter: 100000
        }

    
    }

    fn tabu_search_for_minimum_k_spanning_tree(&mut self, k:u32)->Tree{
        let number_of_vertices: usize = self.graph.get_size();
        let id_random : u32 = self.random_generator.gen_range(1..number_of_vertices ) as u32;
        let initial_vertex = self.graph.get_vertex(id_random);

        //Generate initial solution
        let mut tree : Tree = self.prim_to_k_minimum_tree(initial_vertex,k);
        let mut best : Tree = tree.clone();
        //Get V_in <- V_NH(T^cur_k)
        /*
         * V_NH(T^cur_k) =
         *      { v \in G |
         *          v \notin V(T_k) and
         *          Exist(w) \in V(T_k) where (v,w)\in E(G)
         *      }
         * Our G is K_n so N_NH is the complement of V(T_k).
         */
        self.v_in = self.graph.get_vertex_dictionary().clone();
        for (k,_) in tree.get_tree_vertexs(){
            self.v_in.remove(&k);
        }
        //self.v_out: HashMap<u32, Vertex> = HashMap::new();

        //let mut v_in_vector : Vec<Vertex>= self.v_in.into_values().collect();
        let mut v_in_vector : Vec<Vertex> = Vec::new();
        for (_, v) in self.v_in.iter(){
            v_in_vector.push(*v);
        }

        let mut v_out_vector : Vec<Vertex>= Vec::new();


        let mut tree_k_plus_1 = tree.clone();

        let mut option_vertex_v_in : Option<Vertex> = None;
        let mut option_vertex_v_out : Option<Vertex> = None;

        let mut cont : u32 = 0;

        while self.tt_ten <  self.tt_max && cont < self.max_num_iter {
            cont += 1;
            if v_out_vector.len() == 0 {
                while v_out_vector.len() == 0 {
                    if v_in_vector.len() == 0 {
                        return best;
                    }else{
                        // Sort in decreasing order so we can pop from the back.
                        v_in_vector.sort_unstable_by(
                            |v,w|
                            self.calculate_vertex_edges_average(w, &mut tree)
                            .partial_cmp(
                                & (self.calculate_vertex_edges_average(v, &mut tree))
                            ).unwrap()
                        );

                        //Get vertex_v_in from the back of the vec.
                        let vertex_v_in = v_in_vector.pop().unwrap();
                        option_vertex_v_in = Some(vertex_v_in.clone());
                        //Calculate E_in_1 as the set of edges from vertex_v_in to the tree.
                        let mut e_in_1 = self.graph.get_edges_from_vertex_to_tree(vertex_v_in, &mut tree);

                        //Get e_min_1 from E_in_1
                        e_in_1.sort_unstable_by(
                            |e_1,e_2|
                            e_2.get_weigth().partial_cmp(&e_1.get_weigth()).unwrap()
                        );

                        let mut e_min_1 = e_in_1.pop().unwrap();
                        //Add v_in and e_min_1 to tree_k_plus_1
                        tree_k_plus_1.add_edge_to_tree(e_min_1);

                        while e_in_1.len() != 0{
                            e_min_1 = e_in_1.pop().unwrap();
                            //Add v_in and e_min_1 to tree_k_plus_1
                            let e_min_1_vi = *e_min_1.get_vi();
                            let e_min_1_vf = *e_min_1.get_vf();
                            let mut e_min_1_vertexs_cycle : Vec<Edge> = tree_k_plus_1.get_u_v_unique_path( e_min_1_vi, e_min_1_vf );

                            e_min_1_vertexs_cycle.push(e_min_1);
                            tree_k_plus_1.add_edge_to_tree(e_min_1);


                            //Order from max to min
                            e_min_1_vertexs_cycle.sort_unstable_by(
                                |e_1,e_2|
                                e_1.get_weigth().partial_cmp(&e_2.get_weigth()).unwrap()
                            );
                            let e_max = e_min_1_vertexs_cycle.pop().unwrap();
                            tree_k_plus_1.delete_edge(e_max);

                        }

                        self.v_out = tree.get_tree_vertexs();
                        for (_, v) in self.v_out.iter(){
                            v_out_vector.push(*v);
                        }


                        //break;
                    }
                }
            }else{


                // Sort in increasing order so we can pop from the back.
                //let mut v_out_vector : &Vec<Vertex> = &self.v_out.into_values().collect();
                v_out_vector = Vec::new();
                for (_, v) in self.v_out.iter(){
                    v_out_vector.push(*v);
                }

                v_out_vector.sort_unstable_by(
                    |v,w|
                    self.calculate_vertex_edges_average(v, &mut tree)
                    .partial_cmp(
                        & (self.calculate_vertex_edges_average(w, &mut tree))
                    ).unwrap()
                );

                let vertex_v_out = v_out_vector.pop().unwrap();
                let mut edges_out_min = self.graph.get_edges_from_vertex_to_tree(vertex_v_out, &mut tree);

                //Get e_out_min from edges_out_min
                // Sorting in decreasing order to pop from back
                edges_out_min.sort_unstable_by(
                    |e_1,e_2|
                    e_2.get_weigth().partial_cmp(&e_1.get_weigth()).unwrap()
                );

                let mut e_out_min = edges_out_min.pop().unwrap();

                let f_best = best.get_tree_weigth();
                let f_tree_k_plus_1 = tree_k_plus_1.get_tree_weigth();
                let e_out_min_weigth = e_out_min.get_weigth();

                //############EXPERIMENTAL STUFF###############
                if f_best < f_tree_k_plus_1 - e_out_min_weigth{

                    if self.nic_int > self.nic_max{
                        self.tt_ten = self.tt_min;
                        self.nic_int = 0;
                        break;
                    }else{
                        self.tt_ten = self.tt_ten + self.tt_inc;
                    }

                    continue;
                }
                //###########################################


                let mut connected_components : Vec<Tree> = tree_k_plus_1.get_connected_component_deleting_vertex(option_vertex_v_in.unwrap());
                let mut new_tree_is_created = connected_components.len() == 1 ;

                let mut e_in_2 = self.get_edges_joining_trees(connected_components.clone());
                //Sort in decreasing order to pop from back
                e_in_2.sort_unstable_by(
                    |e1,e2|
                    e1.get_weigth().partial_cmp(&e2.get_weigth()).unwrap()
                );
                let mut unconnected_tree = self.create_unconnected_tree( connected_components.clone() );
                while ! new_tree_is_created {
                    let e_min_2 = e_in_2.pop().unwrap();

                    let f_best = best.get_tree_weigth();
                    let mut f_new_tree = unconnected_tree.get_tree_weigth();
                    let e_min_2_weigth = e_min_2.get_weigth();

                    if f_best < e_min_2_weigth + f_new_tree {
                        self.nic_int = self.nic_int+1;
                        break;
                    }
                    //Check if there's a cycle in new tree if e_min_2 added
                    let e_min_2_vi = e_min_2.get_vi();
                    let e_min_2_vf = e_min_2.get_vf();
                    let posible_cycle = unconnected_tree.get_u_v_unique_path(*e_min_2_vi, *e_min_2_vf);

                    if posible_cycle.len() == 0{
                        unconnected_tree.add_edge_to_tree(e_min_2);
                    }
                    //Check if its a tree
                    let edge_size = unconnected_tree.get_tree_edges().len();
                    let vertex_size = unconnected_tree.get_tree_vertexs().len();

                    if edge_size == vertex_size-1{
                        best = unconnected_tree.clone();
                        new_tree_is_created = true;
                    }

                }

                //break;

            }
        }

        best
    }

    fn create_unconnected_tree( &mut self, components: Vec<Tree> )->Tree{
        let mut new_tree : Tree = Tree::new();
        for mut tree in components{
            for e in &tree.get_tree_edges(){
                new_tree.add_edge_to_tree(*e);
            }
        }
        new_tree
    }

    fn calculate_vertex_edges_average(&mut self, v: &Vertex, tree: &mut Tree)->f64{
        let mut number_of_edges : f64 = 0.0;
        let mut weigth_of_v_to_tree : f64 = 0.0;
        for (w_id,_) in tree.get_tree_vertexs(){

            let v_id_usize : usize = *v.get_id() as usize;
            let w_id_usize : usize = w_id as usize;
            let edge_weight = match self.graph.get_adjacency( v_id_usize, w_id_usize ){
                Adjacency::NonEmpty(e) => e.get_weigth(),
                _ => 0.0
            };
            weigth_of_v_to_tree += edge_weight;
            number_of_edges += 1.0;
        }

        weigth_of_v_to_tree/number_of_edges
    }

    fn prim_to_k_minimum_tree(&mut self, initial_vertex: Vertex, k:u32)->Tree{
        let mut edges_adjacent_to_initial_vertex : Vec<Edge> = self.graph.get_edges_adjacent_to_vertex( initial_vertex);

        edges_adjacent_to_initial_vertex.sort_unstable_by(
            |a,b|
            a.get_weigth().partial_cmp(&b.get_weigth()).unwrap()
        );

        let min_edge: Edge = edges_adjacent_to_initial_vertex[0];

        let mut tree : Tree = Tree::new();
        tree.add_edge_to_tree_keeping_acyclic(min_edge);

        while tree.get_tree_vertexs().len() < (k as usize)  {
            //calculate edges, adyacent to tree.
            let tree_vertexs = tree.get_tree_vertexs();
            let tree_edges = tree.get_tree_edges();

            let mut edges_adyacent_to_tree = self.graph.get_edges_adyacent_to_tree(tree_vertexs, tree_edges);

            edges_adyacent_to_tree.sort_unstable_by(
                |a,b|
                a.get_weigth().partial_cmp(&b.get_weigth()).unwrap()
            );

            let mut edge_added = false;
            let mut j = 0;

            while ! edge_added{
                let min_edge_adyacent_to_tree = edges_adyacent_to_tree[j];
                //Add edge to tree
                edge_added = tree.add_edge_to_tree_keeping_acyclic(min_edge_adyacent_to_tree);
                j += 1;
            }
        }

        tree
    }

    fn get_edges_joining_trees( &mut self, trees: Vec<Tree> )->Vec<Edge>{

        let mut edges_joining_trees : Vec<Edge> = Vec::new();

        let number_of_trees = trees.len();
        for i in 0..number_of_trees-1{
            let mut tree_a = trees[i].clone();
            for j in (i+1)..number_of_trees{
                let mut tree_b = trees[j].clone();

                for (_ , v_tree_a ) in &tree_a.get_tree_vertexs(){
                    let mut edges_1 = self.graph.get_edges_from_vertex_to_tree( *v_tree_a, &mut tree_b );
                    edges_joining_trees.append(&mut edges_1);
                }
            }
        }

        edges_joining_trees
    }
    fn set_random_value(&mut self, new_random: StdRng){
        self.random_generator = new_random
    }

}

fn read_lines<P>(file_name: P) ->
    io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>,
{
    let file = File::open(file_name)?;
    Ok(io::BufReader::new(file).lines())
}
fn get_list_of_points_in_file(file_name: &str)->Vec<Vertex>
{
    let mut vertexs :Vec<Vertex> = Vec::new();
    if let Ok(lines) = read_lines(file_name){
        let mut id_cont = 1;
        for line in lines{
            if let Ok(x_y_pair) = line {
                let x_y : Vec<&str> = x_y_pair.split(",").collect();
                let x = match f64::from_str( x_y[0] ){
                    Ok(x_val) => x_val,
                    Err(_)=>{
                        println!("Error while parsing point: {}", x_y_pair);
                        exit(1);
                    }
                };
                let y = match f64::from_str( x_y[1] ){
                    Ok(y_val) => y_val,
                    Err(_)=>{
                        println!("Error while parsing point: {}", x_y_pair);
                        exit(1);
                    }
                };
                vertexs.push(
                    Vertex{
                        id : id_cont,
                        x : x,
                        y: y
                    }
                );
                id_cont += 1;
            }
        }
    }
    vertexs
}

fn print_use(){
    let s = format!(
        "   SYSTEM USE:
            $ cargo run --release FILE_NAME K OPTION NUM
                -> FILE_NAME : Name of the file with format specified.
                -> K: number of vertexs of the k-tree
                -> OPTION:
                        => Use 'eval' to evaluate a seed in NUM, NUM is a number
                        => Use 'expr' to experiment with NUM diferent seeds.
        "
    );
    println!("{}",s);
}

fn main() {

    let args : Vec<String> = env::args().collect();
    let number_of_params = args.len();

    if number_of_params < 5{
        print_use();
        exit(1);
    }

    let arch = &args[1];
    let k : u32 =  match args[2].parse::<u32>(){
        Result::Ok(kk) => kk,
        _ =>{
            println!("Error in K parameter");
            print_use();
            exit(1);
        }
    };
    let option  = args[3].clone();

    if !option.eq("eval") && !option.eq("expr") {
        print_use();
        exit(1);
    }

    let seed_or_iterations : u64 =  match args[4].parse::<u64>(){
        Result::Ok(v) => v,
        _ =>{
            println!("Error in NUM parameter");
            print_use();
            exit(1);
        }
    };

    //------------INITIALIZE GRAPH AND HEURISTIC-------------

    let vertexs :Vec<Vertex> = get_list_of_points_in_file(&arch);
    let mut g: Graph = Graph::new(vertexs.clone());
    let mut problem = MinimumKTreeHeuristic::new(
        g.clone(),
        StdRng::seed_from_u64(0),
        vertexs.len() as u32,
        k
    );
    //-------------------------------------------------------

    if option.eq("eval"){

        let seed = StdRng::seed_from_u64(seed_or_iterations);
        problem.set_random_value(seed);

        let mut final_tree: Tree = problem.tabu_search_for_minimum_k_spanning_tree(k);
        let mut wf = 0.0;
        for e in final_tree.get_tree_edges(){
            wf += e.get_weigth();
        }
        println!("FINAL TREE:\n{:?}", final_tree);
        println!("WEIGTH OF TREE: {}", wf);
        println!("SEED: {}", seed_or_iterations);
        println!("SALIDA PUNTOS E: ");

        for e in final_tree.get_tree_edges(){
            println!("{},{},{},{}",
                     e.get_vi().get_x(),
                     e.get_vi().get_y(),
                     e.get_vf().get_x(),
                     e.get_vf().get_y()
            );
        }
    }

    if option.eq("expr"){
        for s in 1..(seed_or_iterations as usize){

            let seed = StdRng::seed_from_u64(s as u64);

            problem = MinimumKTreeHeuristic::new(
                g.clone(),
                seed,
                vertexs.len() as u32,
                k
            );

            let mut final_tree: Tree = problem.tabu_search_for_minimum_k_spanning_tree(k);

            let mut wf = 0.0;
            for e in final_tree.get_tree_edges(){
                wf += e.get_weigth();
            }

            println!("FINAL TREE:\n{:?}", final_tree);
            println!("WEIGTH OF TREE: {}", wf);
            println!("SEED: {}", s);
            println!("SALIDA PUNTOS E: ");

            for e in final_tree.get_tree_edges(){
                println!("{},{},{},{}",
                        e.get_vi().get_x(),
                        e.get_vi().get_y(),
                        e.get_vf().get_x(),
                        e.get_vf().get_y()
                );
            }
        }
    }

}
