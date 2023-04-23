use std::process::exit;
use std::fs::File;
use std::io::{self,BufRead};
use std::path::Path;

use std::collections::HashMap;
use std::str::FromStr;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::Entry;
use std::collections::VecDeque;
//use priority_queue::DoublePriorityQueue;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

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
            //println!("i, {:?}", vi);
            //println!("{:?}", vi);
            for j in (i+1)..size{
                //if i != j{
                    let vf : &Vertex = &vertexs_list[j];
                    //println!("j, {:?}", vf);

                    let id_vi : usize = *vi.get_id() as usize;
                    let id_vf : usize = *vf.get_id() as usize;

                    let index_a : usize = (id_vi*v_size + id_vf) as usize;
                    let index_b : usize = (id_vf*v_size + id_vi) as usize;

                    let e : Edge = Edge{ vi:*vi, vf: *vf };

                    adjacent_vector[index_a] = Adjacency::NonEmpty( e );
                    adjacent_vector[index_b] = Adjacency::NonEmpty( e );
                //}
            }
        }

        for i in 0..v_size{
            for j in 0..v_size{
                println!("i:{}, j:{}, val:{:?}",i, j, adjacent_vector[i*v_size+j]);
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
            //println!("{}",c.clone());
            //println!("{:?}",self.adjacencies_matrix[i] );
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
            //println!("i:{}",i);
            for j in (i+1)..self.size{
            //println!("j::{}",j);
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

    fn get_u_v_unique_path( &mut self, u: Vertex, v: Vertex )->Vec<Edge>{

        let v_id = v.get_id();
        let u_id = u.get_id();

        //First we build our neighbour tree from vertex
        let mut vertex_neighbours : HashMap<u32, Vec<Vertex> > = HashMap::new();
        let mut cycle : Vec<Edge> = Vec::new();
        let mut pre_path : Vec<Vertex> = Vec::new();
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
        //Find unique trayectory. DFS because is nicer for tree
        let mut queue_dfs: VecDeque<Vertex> = VecDeque::new();
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
        queue_dfs.push_front(v);

        let mut tray_found: bool = false;

        while queue_dfs.len() != 0 && !tray_found{
            let w : Vertex = queue_dfs.pop_front().unwrap();

            tray_found = w.eq(&u);

            for r  in vertex_neighbours[w.get_id()].clone(){

                if ! visited_register.get(r.get_id()).unwrap() {

                    *visited_register.get_mut(r.get_id()).unwrap() = true;
                    parents.insert(*r.get_id(), *w.get_id() );
                    //*parents_v.get_mut(r.get_id()).unwrap() = w;
                    parents_v.insert(*r.get_id(), w);
                    //parents[r.get_id()] = w;
                    queue_dfs.push_back(r);
                }
            }
        }
        //Build tray
        println!("PRE-CICLO-MAP: {:?}", parents);

        let mut tray: Vec<Edge> = Vec::new();
        let mut first : Vertex = u;
        let mut next: Vertex = *parents_v.get(first.get_id()).unwrap();
        tray.push(Edge{vi: first, vf: next});
        while( !v.eq( &next) ){
            first = next;
            next = *parents_v.get(first.get_id()).unwrap();
            let e = Edge{vi: first, vf: next};
            println!("crea: {:?}", e);
            tray.push(e);
        }

        tray
    }

    /*
    fn get_cycle_edges( &mut self , number_of_vertex: usize)->Vec<Edge>{
        //First we build our neighbour tree from vertex
        let mut vertex_neighbours : HashMap<u32, Vec<Vertex> > = HashMap::new();
        let mut cycle : Vec<Edge> = Vec::new();
        let mut pre_cycle : Vec<Vertex> = Vec::new();
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
    //println!("MAPAAAAAAA: {:?}", vertex_neighbours);

        //We search for a tree in that vertex
        let mut visited_register : Vec<bool> = vec![false; number_of_vertex];
        for (v_id, _) in self.tree_vertex_dictionary.clone(){
            if !visited_register[v_id as usize] {
                self.sub_detect_cyclic(&vertex_neighbours, &mut pre_cycle, v_id, &mut visited_register, 0);

                println!("PRE_CICLOOO: {:?}", pre_cycle);
            }
        }
        //Create final cicle
        // Ineficient way to verify there exists such edges
        for i in 1..(pre_cycle.len()){
            /*
            let e = self.find_edge_by_vertices(
                pre_cycle[i],
                pre_cycle[i+1]
            );
            */
            let e = Edge{vi: pre_cycle[i-1], vf: pre_cycle[i]};
            cycle.push(e);
        }
        cycle.push(Edge{vi:pre_cycle[0], vf:pre_cycle[pre_cycle.len()-1]});
        cycle
    }


    fn sub_detect_cyclic( &mut self, adjacencies: &HashMap<u32,v_id: u32, visited_register: &mut Vec<bool>, v_parent: u32 )-> Option<Vec<Vertex>>{

        //Mark vertex as visited
        visited_register[v_id as usize] = true;
        let neighbors = adjacencies.get(&v_id).unwrap().clone();
        println!("Empezando en: {:?}, con padre; {:?}", v_id, v_parent);
        println!("Vecinos:{:?}", neighbors);
        println!("visitados:{:?}", visited_register);
        for w in neighbors{
            let w_id = w.get_id();

            if !visited_register[*w_id as usize] {
               let sub_path = self.sub_detect_cyclic( adjacencies, *w_id, visited_register, v_id);

                if sub_path.is_some() {
                    return sub_path;
                }
            }else if v_parent != *w_id{
                    return Some(vec![w]);
            }
        }
        None
    }
    */

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
            println!("ELIMINAAAAA ENTRAA");
            println!("Lista edges tree: {:?}", self);
            println!("A eliminar: {:?}", e);

        if self.tree_edge_list.contains(&e){
            let index =  self.tree_edge_list.iter().position(|x| x.eq(&e) ).unwrap();
            self.tree_edge_list.remove(index);
            println!("ELIMINAAAAA");

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
    random_generator: StdRng
}

impl MinimumKTreeHeuristic{

    fn tabu_search_for_minimum_k_spanning_tree(&mut self, k:u32)->Tree{
        let number_of_vertices: usize = self.graph.get_size();
        let id_random : u32 = self.random_generator.gen_range(1..number_of_vertices ) as u32;
        let initial_vertex = self.graph.get_vertex(id_random);

        //Generate initial solution
        let mut tree : Tree = self.prim_to_k_minimum_tree(initial_vertex,k);
        let mut best : Tree = tree.clone();
        ///*
        for e in tree.get_tree_edges(){
            println!("{:?}",e);

;        }
        //*/
        //Get V_in <- V_NH(T^cur_k)
        /*
         * V_NH(T^cur_k) =
         *      { v \in G |
         *          v \notin V(T_k) and
         *          Exist(w) \in V(T_k) where (v,w)\in E(G)
         *      }
         * Our G is K_n so N_NH is the complement of V(T_k).
         */
        let mut v_in = self.graph.get_vertex_dictionary().clone();
        for (k,_) in tree.get_tree_vertexs(){
            v_in.remove(&k);
        }
        println!("{:?}", v_in);
        let mut v_in_vector : Vec<Vertex>= v_in.into_values().collect();

        let mut v_out: HashMap<u32, Vertex> = HashMap::new();

        let mut tree_k_plus_1 = tree.clone();
        //println!("COSA ANTESSSSS: {:?}", tree_k_plus_1.get_cycle_edges(self.graph.get_size()));

        let vertex_in : Vertex;
        let vertex_out : Vertex;

        ///*
        loop{
            if v_out.len() == 0 {
                while v_out.len() == 0 {
                    if v_in_vector.len() == 0 {
                        return best;
                    }else{
                        //let mut v_in_ordered : Vec<Vertex>= v_in.into_values().collect();
                        // Sort in decreasing order so we can pop from the back.
                        v_in_vector.sort_unstable_by(
                            |v,w|
                            self.calculate_vertex_edges_average(w, &mut tree)
                            .partial_cmp(
                                & (self.calculate_vertex_edges_average(v, &mut tree))
                            ).unwrap()
                        );

                        println!("{:?}", v_in_vector);
                        //Get vertex_v_in from the back of the vec.
                        let vertex_v_in = v_in_vector.pop().unwrap();
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
                            println!("ANTES QUITANDO Y METIENEDO ARISTAS{:?}", tree_k_plus_1);

                        while e_in_1.len() != 0{
                            e_min_1 = e_in_1.pop().unwrap();
                            //Add v_in and e_min_1 to tree_k_plus_1
                            let e_min_1_vi = *e_min_1.get_vi();
                            let e_min_1_vf = *e_min_1.get_vf();
                            let mut e_min_1_vertexs_cycle : Vec<Edge> = tree_k_plus_1.get_u_v_unique_path( e_min_1_vi, e_min_1_vf );

                            e_min_1_vertexs_cycle.push(e_min_1);
                            tree_k_plus_1.add_edge_to_tree(e_min_1);

                            println!("INSERTA E: {:?}", e_min_1);
                            println!("CICLOOO: {:?}", e_min_1_vertexs_cycle);

                            //Order from max to min
                            e_min_1_vertexs_cycle.sort_unstable_by(
                                |e_1,e_2|
                                e_1.get_weigth().partial_cmp(&e_2.get_weigth()).unwrap()
                            );
                            let e_max = e_min_1_vertexs_cycle.pop().unwrap();
                            tree_k_plus_1.delete_edge(e_max);
                            println!("QUITANDO Y METIENEDO ARISTAS: {:?}\n{:?}",e_max, tree_k_plus_1);

                        }

                        v_out = tree.get_tree_vertexs();

                        println!("T_cur_k: {:?}", tree);
                        println!("T_k_plus_1: {:?}", tree_k_plus_1);

                        break;

                    }
                }
            }
        }

        //*/
        best
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

        println!("{}",weigth_of_v_to_tree/number_of_edges);
        weigth_of_v_to_tree/number_of_edges
    }

    fn prim_to_k_minimum_tree(&mut self, initial_vertex: Vertex, k:u32)->Tree{
        println!("EXCECUTING PRIM:::");
        println!("Initial vertex: {:?}", initial_vertex);
        let mut edges_adjacent_to_initial_vertex : Vec<Edge> = self.graph.get_edges_adjacent_to_vertex( initial_vertex);
        //aristas_mayor_a_menor.sort_unstable_by(|a,b| b.partial_cmp(a).unwrap() );
        edges_adjacent_to_initial_vertex.sort_unstable_by(
            |a,b|
            a.get_weigth().partial_cmp(&b.get_weigth()).unwrap()
        );
        /*
        for i in &edges_adjacent_to_initial_vertex{
            println!("e:{:?}, peso:{}",i,i.get_weigth());
        }
        */
       // println!("{:?}, \n tam: {}", edges_adjacent_to_initial_vertex, edges_adjacent_to_initial_vertex.len());
        let min_edge: Edge = edges_adjacent_to_initial_vertex[0];
        println!("initial edge: {:?}",min_edge);
        let mut tree : Tree = Tree::new();
        tree.add_edge_to_tree_keeping_acyclic(min_edge);
        while tree.get_tree_vertexs().len() < (k as usize)  {
            //calculate edges, adyacent to tree.
            let tree_vertexs = tree.get_tree_vertexs();
            let tree_edges = tree.get_tree_edges();

            let mut edges_adyacent_to_tree = self.graph.get_edges_adyacent_to_tree(tree_vertexs, tree_edges);
            println!("{:?}", edges_adyacent_to_tree);
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
                //println!("{}", x_y_pair);
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

fn main() {
    //let arch : &str = "puntosCanekBueno.txt";
    let arch : &str = "puntos.txt";
    let k = 4;

    let vertexs :Vec<Vertex> = get_list_of_points_in_file(arch);
    let g: Graph = Graph::new(vertexs.clone());
    let seed = StdRng::seed_from_u64(0);
    println!("{:?}", vertexs);
    let mut problem = MinimumKTreeHeuristic{graph: g, random_generator: seed};
    problem.tabu_search_for_minimum_k_spanning_tree(k);
    println!("Hello, world!");
}
