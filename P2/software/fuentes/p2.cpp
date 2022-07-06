#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <limits>
#include <cassert>
#include <unordered_map>
#include <set>
#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

default_random_engine generator;
int num_feats;
const int K = 5;
const double alpha = 0.5;

// Elemento muestral
struct SampleElement {
  vector<double> features;
  string label;
};

// Cromosoma; genes y valor de f. objetivo
struct Chromosome{
  vector<double> genes;
  double obj;
};

bool operator <(const Chromosome& c1, const Chromosome& c2){
  return c1.obj < c2.obj;
}
bool operator >(const Chromosome& c1, const Chromosome& c2){
  return c1.obj > c2.obj;
}

// Lee el fichero de datos
void read_arff(string filename, vector<SampleElement>& sample_els){
	ifstream file(filename);
	string line;

	if (file){
		// Leer hasta llegar a los datos
		do{
			file >> line;
		}while(line.compare("@data")!=0);

		file >> line; // leer primer elemento

		// Mientras haya elementos por leer
		do{
			SampleElement element;
			string val;
			double feature;
			// leer valores atributos
			for (int i = 0; i < line.size(); i++){
				if (line[i] != ',')
					val = val + line[i];
				else{
					feature = atof(val.c_str());
					element.features.push_back(feature);
					val.clear();
				}
			}

			element.label = val;

			sample_els.push_back(element);

			file >> line;

		}while(!file.eof());

		num_feats=sample_els[0].features.size();

		file.close();
	}
	else
		cout << "File error" << endl;
}

// Normalización de datos
void normalization(vector<SampleElement>& sample_els){
	vector<double> max(num_feats, -numeric_limits<double>::max());
	vector<double> min(num_feats, numeric_limits<double>::max());
	double feat_val;

	for (auto it = sample_els.begin(); it != sample_els.end(); ++it){
		for (int j=0; j < num_feats; j++){
			feat_val = (*it).features[j];
			if (feat_val > max[j])
				max[j] = feat_val;
			else
        if (feat_val < min[j])
				    min[j] = feat_val;
		}
	}

	double max_min;
	for (int k=0; k<sample_els.size(); k++)
		for (int j=0; j < num_feats; j++){
			max_min = max[j]-min[j];
			if (max_min > 0)
				sample_els[k].features[j] = (sample_els[k].features[j]-min[j]) / (max_min);
		}
}

vector<vector<SampleElement>> make_k_folds(const vector<SampleElement>& sample_els){
	unordered_map<string, int> label_indexer;
  vector<vector<SampleElement>> partitions(K, vector<SampleElement>());

  int count;
  for (auto it = sample_els.begin(); it != sample_els.end(); ++it) {
    count = label_indexer[(*it).label];
    partitions[count].push_back(*it);
    label_indexer[(*it).label] = (label_indexer[(*it).label] + 1) % K;
  }

  return partitions;
}

/*
    1-NN
*/


double euclidean_distance2(const vector<double>& a, const vector<double>& b){
	double dist = 0;
  int a_s = a.size();
  assert(a_s == b.size());

	for (int i=0; i < a_s; i++){
		dist += (a[i]-b[i]) * (a[i]-b[i]);
	}

	return dist;
}

double euclidean_distance2_w(const vector<double>& a, const vector<double>& b, const vector<double>& weights){
	double dist = 0;
  int a_s = a.size();
  assert(a_s == b.size());

	for (int i=0; i < a_s; i++)
    if(weights[i] >= 0.2) // se descartan las características con peso < 2
		  dist += weights[i] * (a[i]-b[i]) * (a[i]-b[i]);

	return dist;
}

// 1NN leave one out
string one_NN_lo(SampleElement sam_el, const vector<SampleElement>& sam_elements, vector<double> weights, int leave_out){

	string min_l;
	double min_dist = numeric_limits<double>::max();
  double d;

	for (int i = 0; i < sam_elements.size(); i++){
		if (i != leave_out){
			d = euclidean_distance2_w(sam_elements[i].features, sam_el.features, weights);
			if (d < min_dist){
				min_l = sam_elements[i].label;
				min_dist = d;
			}
		}
	}

	return min_l;
}

string one_NN(SampleElement sam_el, const vector<SampleElement>& sam_elements, vector<double> weights){

	string min_l;
	double min_dist = numeric_limits<double>::max();
  double d;

	for (auto it = sam_elements.begin(); it != sam_elements.end(); ++it){
			d = euclidean_distance2_w((*it).features, sam_el.features, weights);
			if (d < min_dist){
				min_l = (*it).label;
				min_dist = d;
			}
	}

	return min_l;
}

/*
    OBJETIVE FUNCTION
*/

double class_rate(const vector<string>& class_labels, const vector<SampleElement>& test){
	int n=0;
  int class_labels_s = class_labels.size();

	for (int i=0; i < class_labels_s; i++)
    if (class_labels[i] == test[i].label)
      n++;

	return 100.0*n / class_labels_s;
}

double red_rate(const vector<double>& weights){
	int feats_reduced = 0;

	for (auto it = weights.begin(); it != weights.end(); ++it)
		if (*it < 0.2)
      feats_reduced++;

	return 100.0 * feats_reduced / num_feats;
}

double obj_function(double class_rate, double red_rate){
	return alpha*class_rate + (1.0-alpha)*red_rate;
}


/*
  Mutación
*/
void mutation(vector<double>& weights, int i, normal_distribution<double>& n_dist){
	weights[i] += n_dist(generator);

	if (weights[i] > 1)
    weights[i] = 1;
	else {
    if (weights[i] < 0)
      weights[i] = 0;
  }
}

/*
  Operador de selección
  Torneo binario con reemplazamiento
*/
Chromosome selection(const vector<Chromosome>& population){
  int pop_size_1 = population.size()-1;

  Chromosome c1 = population[Random::get(0,pop_size_1)];
  Chromosome c2 = population[Random::get(0,pop_size_1)];

  if (c1.obj > c2.obj)
    return c1;
  else
    return c2;
}
/*
  Operador de mutación
*/
void mutate_pop(vector<Chromosome>& population, int num_muts){
  normal_distribution<double> normal_dist(0.0, 0.3);
  set<pair<int, int>> mutated;

  int chr_i, gen_i;

  int pop_size_1 = population.size() - 1;
  int genes_size_1 = num_feats - 1;

  int muts = 0;
  while (muts < num_muts){
    // se toma un cromosoma y gen aleatorios
    chr_i = Random::get(0, pop_size_1);
    gen_i = Random::get(0, genes_size_1);
    // si aún no se ha mutado ese gen de ese cromosoma
    if (mutated.find(make_pair(chr_i,gen_i)) == mutated.end()){
      //mutamos
      mutation(population[chr_i].genes, gen_i, normal_dist);

      muts++;
      mutated.insert(make_pair(chr_i, gen_i));
      population[chr_i].obj = -1.0;
    }
  }

}


void obj_to_chromosome(Chromosome& c, const vector<SampleElement>& training){
  vector<string> class_labels;

  for (unsigned i = 0; i < training.size(); i++)
    class_labels.push_back( one_NN_lo(training[i], training, c.genes, i) );

  c.obj = obj_function(class_rate(class_labels, training), red_rate(c.genes));
}

/*
  Cruce BLX
*/

pair<Chromosome, Chromosome> blx_cross(const Chromosome &p1, const Chromosome &p2) {
  Chromosome d1, d2; // descendientes
  int pgenes_size = p1.genes.size();

  d1.genes.resize(pgenes_size);
  d2.genes.resize(pgenes_size);

  double up_b, low_b;
  double cmin, cmax, i_a;
  for (int i = 0; i < pgenes_size; i++){
    if (p1.genes[i] > p2.genes[i]){
      cmin = p2.genes[i];
      cmax = p1.genes[i];
    }
    else{
      cmin = p1.genes[i];
      cmax = p2.genes[i];
    }

    i_a = (cmax - cmin) * 0.3;

    low_b = cmin - i_a;
    up_b = cmax + i_a;
    d1.genes[i] = Random::get(low_b, up_b);
    d2.genes[i] = Random::get(low_b, up_b);

    // restringimos a [0,1]
    if (d1.genes[i] < 0.0)
      d1.genes[i] = 0.0;
    else if (d1.genes[i] > 1.0)
      d1.genes[i] = 1.0;
    if (d2.genes[i] < 0.0)
      d2.genes[i] = 0.0;
    else if (d2.genes[i] > 1.0)
      d2.genes[i] = 1.0;
  }
  // para evitar evaluaciones de f.obj redundantes antes de la mutación
  d1.obj = -1.0;
  d2.obj = -1.0;

  return make_pair(d1, d2);
}

pair<Chromosome, Chromosome> arith_cross(const Chromosome& p1, const Chromosome& p2) {
  Chromosome d1, d2; // descendientes
  int gen_size = p1.genes.size();
  double alpha = Random::get(0.01, 0.99);

  d1.genes.resize(gen_size);
  d2.genes.resize(gen_size);

  for (int i = 0; i < gen_size; i++){
    d1.genes[i] = alpha * p1.genes[i] + (1 - alpha) * p2.genes[i];
  }

  alpha = Random::get(0.01, 0.99);
  for (int i = 0; i < gen_size; i++){
    d2.genes[i] = alpha * p1.genes[i] + (1 - alpha) * p2.genes[i];
  }
  // para evitar evaluaciones de f.obj redundantes antes de la mutación
  d1.obj = -1.0;
  d2.obj = -1.0;

  return make_pair(d1, d2);
}

pair<Chromosome, Chromosome> sbx2_cross(const Chromosome& p1, const Chromosome& p2) {
  Chromosome d1, d2; // descendientes
  int gen_size = p1.genes.size();
  d1.genes.resize(gen_size);
  d2.genes.resize(gen_size);
  double nu = 2;

  double u = Random::get(0.0001, 0.9999);
  double b;

  if (u < 0.5){
    b = pow(2*u, 1/(nu+1));
  }
  else{
    b = pow(1/(2*(1-u)), 1/(nu+1));
  }

  for (int i = 0; i < gen_size; i++){
    d1.genes[i] = 0.5 * ( (1+b)*p1.genes[i] + (1-b)*p2.genes[i] );
    d2.genes[i] = 0.5 * ( (1-b)*p1.genes[i] + (1+b)*p2.genes[i] );
  }

  // para evitar evaluaciones de f.obj redundantes antes de la mutación
  d1.obj = -1.0;
  d2.obj = -1.0;

  return make_pair(d1, d2);
}

/*
  AGG
  cross_type = 0 => blx_cross
  cross_type = 1 => arith_cross
*/
vector<double> agg(const vector<SampleElement>& training, int cross_type){
  vector<Chromosome> population, inter_population;
  Chromosome best_parent, best_descendent;
  int pop_size = 30;
  int exp_num_crosses = round(0.7 * pop_size / 2.0); // núm. esperado cruces (prob cruce * población / 2)
  int chr_to_cross = 2 * exp_num_crosses; // núm. chromosomas que participan en cruce
  int exp_num_muts = round(0.1 * pop_size); // número mutaciones (prob. mut. de cromosoma por total de chromosomas)
  int evals = 0;

  pair<Chromosome, Chromosome> (*cross) (const Chromosome&, const Chromosome&);
  if (cross_type == 0)
    cross = blx_cross;
  else if (cross_type == 1)
    cross = arith_cross;
  else if (cross_type == 2)
    cross = sbx2_cross;

  population.resize(pop_size);
  // genera una población inicial aleatoria de 30 cromosomas
  for (int i = 0; i < pop_size; i++){
    for (int j = 0; j < num_feats; j++){
      population[i].genes.push_back(Random::get(0.0,1.0));
    }
    // se obtiene valor función objetivo en cada cromosoma
    obj_to_chromosome(population[i], training);
    evals++;
  }
  // Se guarda el mejor cromosoma, el mejor padre
  best_parent = *max_element(population.begin(), population.end());

  inter_population.resize(pop_size);
  pair<Chromosome, Chromosome> descendents;
  Chromosome sel1, sel2;
  while (evals < 15000){
    //selecciona población intermedia que se cruza
    for (int i = 0; i < chr_to_cross; i+=2){
      sel1 = selection(population);
      sel2 = selection(population);
      descendents = cross(sel1, sel2);

      inter_population[i] = descendents.first;
      inter_population[i+1] = descendents.second;
    }
    // selecciona la población intermedia que no se cruza
    for (int i = chr_to_cross; i < pop_size ; i++){
      inter_population[i] = selection(population);
    }

    // hace exp_num_muts mutaciones en la población intermedia
    mutate_pop(inter_population, exp_num_muts);

    // se reemplaza la población
    population = inter_population;

    // se evalua f.obj para los cruzados y mutados que no tienen valor de f.obj
    for (Chromosome& c : population){
      if (c.obj == -1){
        obj_to_chromosome(c, training);
        evals++;
      }
    }

    best_descendent = *max_element(population.begin(), population.end());

    // se mantiene el mejor de P(t) (elitismo)
    if (best_parent > best_descendent)
      *min_element(population.begin(), population.end()) = best_parent;
    else
      best_parent = best_descendent;

  }

  // se devuelven los genes del mejor cromosoma encontrado
  return max_element(population.begin(), population.end())->genes;
}

/*
  AGE
*/
vector<double> age(const vector<SampleElement>& training, int cross_type){
  vector<Chromosome> population, inter_population;
  Chromosome best_parent, best_descendent;
  int pop_size = 30;
  int evals = 0;
  normal_distribution<double> normal_dist(0.0, 0.3);

  pair<Chromosome, Chromosome> (*cross) (const Chromosome&, const Chromosome&);
  if (cross_type == 0)
    cross = blx_cross;
  else if (cross_type == 1)
    cross = arith_cross;
  else if (cross_type == 2)
    cross = sbx2_cross;

  population.resize(pop_size);
  // genera una población inicial aleatoria de 30 cromosomas
  for(int i = 0; i < pop_size; i++){
    for(int j = 0; j < num_feats; j++){
      population[i].genes.push_back(Random::get(0.0,1.0));
    }
    // se obtiene valor función objetivo en cada cromosoma
    obj_to_chromosome(population[i], training);
    evals++;
  }

  vector<Chromosome>::iterator min_c;
  inter_population.resize(2);
  Chromosome sel1, sel2;
  pair<Chromosome, Chromosome>  descendents;
  while (evals < 15000){
    //selecciona población intermedia que se cruza
    sel1 = selection(population);
    sel2 = selection(population);
    descendents = cross(sel1, sel2);
    inter_population[0] = descendents.first;
    inter_population[1] = descendents.second;

    // mutaciones en la población intermedia, prob. mutación individuo de 0.1
    for (int i = 0; i < 2; i++){
      if (Random::get(0.0, 1.0) <= 0.1){
        mutation(inter_population[i].genes, Random::get(0, num_feats-1), normal_dist);
        inter_population[i].obj = -1.0;
      }
    }

    // se evalúa f.obj población pob_intermedia
    for (Chromosome& c : inter_population){
      if (c.obj == -1){
        obj_to_chromosome(c, training);
        evals++;
      }
    }

    // sustituir peores elementos de la población por los de pob. intermedia si son mejores
    for (Chromosome& c : inter_population){
      min_c = min_element(population.begin(), population.end());
      if (*min_c < c)
        *min_c = c;
    }

  }

  // se devuelven los genes del mejor cromosoma encontrado
  return max_element(population.begin(), population.end())->genes;
}

/*
  low intensity local search
*/
void li_ls(const vector<SampleElement>& training, Chromosome& c, int& obj_eval_count) {
  normal_distribution<double> n_dist(0.0, 0.3);
  vector<int> comp_indexes;
  int n = num_feats;
  double best_obj;
  int gen_neighbours = 0;
  bool new_best_obj = false;
  int mod_pos = 0;
  int max_objf_evals = 15000;

  // se inicializan índices de las componentes
  for (int i = 0; i < n; i++)
    comp_indexes.push_back(i);

  best_obj = c.obj;

  Chromosome muted_c;
  int comp_to_mut;

  while ((gen_neighbours < 2*n)  && (obj_eval_count < max_objf_evals)) {
    // aleatorizamos componentes a mutar si se han recorrido todas (o en primera iteración)
    if (new_best_obj || (mod_pos % num_feats == 0)){
      new_best_obj = false;
      shuffle(comp_indexes.begin(), comp_indexes.end(), generator);
      mod_pos = 0;
    }
    comp_to_mut = comp_indexes[gen_neighbours % n]; // componente a mutar

    muted_c = c;
    mutation(muted_c.genes, comp_to_mut, n_dist); // mutación
    gen_neighbours++;

    // se asigna f. obj a cromosoma con los pesos mutados (con leave one out)
    obj_to_chromosome(muted_c, training);

    // si se ha mejorado, actualizamos mejor objetivo, cromosoma y vecinos generados
    if (muted_c.obj > best_obj) {
      c = muted_c;
      best_obj = c.obj;

      new_best_obj = true;
    }

    obj_eval_count++;
    mod_pos++;
  }

}

vector<double> am_1010(const vector<SampleElement>& training, int cross_type){
  vector < Chromosome > population, inter_population;
  Chromosome best_parent, best_descendent;
  int pop_size = 10;
  int exp_num_crosses = round(0.7 * pop_size / 2.0); // núm. esperado cruces (prob cruce * población)
  int chr_to_cross = 2*exp_num_crosses; // núm. chromosomas que participan en cruce
  int exp_num_muts = 0.1 * pop_size; // número mutaciones (prob. mut. de cromosoma por total de chromosomas)
  int evals = 0;
  int generation = 1;

  pair<Chromosome, Chromosome> (*cross) (const Chromosome&, const Chromosome&);
  if (cross_type == 0)
    cross = blx_cross;
  else if (cross_type == 1)
    cross = sbx2_cross;

  population.resize(pop_size);
  // genera una población inicial aleatoria de 30 cromosomas
  for(int i = 0; i < pop_size; i++){
    for(int j = 0; j < num_feats; j++){
      population[i].genes.push_back(Random::get(0.0,1.0));
    }
    // se obtiene valor función objetivo en cada cromosoma
    obj_to_chromosome(population[i], training);
    evals++;
  }

  inter_population.resize(pop_size);
  pair<Chromosome, Chromosome> descendents;

  while (evals < 15000){
    // Se guarda el mejor cromosoma, el mejor padre
    best_parent = *max_element(population.begin(), population.end());

    // selecciona población intermedia que se cruza
    Chromosome sel1, sel2;
    for (int i = 0; i < chr_to_cross; i+=2){
      sel1 = selection(population);
      sel2 = selection(population);
      descendents = blx_cross(sel1, sel2);

      inter_population[i]=descendents.first;
      inter_population[i+1]=descendents.second;
    }
    //selecciona la población intermedia que no se cruza
    for (int i = chr_to_cross; i < pop_size ; i++){
      inter_population[i] = selection(population);
    }

    // hace exp_num_muts mutaciones en la población intermedia
    mutate_pop(inter_population, exp_num_muts);

    // se reemplaza la población
    population = inter_population;

    // se evalua f.obj para los cruzados y mutados que no tienen valor de f.obj
    for (Chromosome& c : population){
      if (c.obj == -1){
        obj_to_chromosome(c, training);
        evals++;
      }
    }

    best_descendent = *max_element(population.begin(), population.end());

    // se mantiene el mejor de P(t) (elitismo)
    if (best_parent > best_descendent)
      *min_element(population.begin(), population.end()) = best_parent;

    if (generation % 10 == 0){
      for (Chromosome& c : population){
        li_ls(training, c, evals);
      }
    }

    generation++;
  }

  // se devuelven los genes del mejor cromosoma encontrado
  return max_element(population.begin(), population.end())->genes;
}

vector<double> am_1001(const vector<SampleElement>& training, int cross_type){
  vector < Chromosome > population, inter_population;
  Chromosome best_parent, best_descendent;
  int pop_size = 10;
  int exp_num_crosses = round(0.7 * pop_size / 2.0); // núm. esperado cruces (prob cruce * población)
  int chr_to_cross = 2*exp_num_crosses; // núm. chromosomas que participan en cruce
  int exp_num_muts = 0.1 * pop_size;// número mutaciones (prob. mut. de cromosoma por total de chromosomas)
  int evals = 0;
  int generation = 1;

  pair<Chromosome, Chromosome> (*cross) (const Chromosome&, const Chromosome&);
  if (cross_type == 0)
    cross = blx_cross;
  else if (cross_type == 1)
    cross = sbx2_cross;

  population.resize(pop_size);
  // genera una población inicial aleatoria de 30 cromosomas
  for(int i = 0; i < pop_size; i++){
    for(int j = 0; j < num_feats; j++){
      population[i].genes.push_back(Random::get(0.0,1.0));
    }
    // se obtiene valor función objetivo en cada cromosoma
    obj_to_chromosome(population[i], training);
    evals++;
  }

  inter_population.resize(pop_size);
  pair<Chromosome, Chromosome> descendents;

  while (evals < 15000){
    // Se guarda el mejor cromosoma, el mejor padre
    best_parent = *max_element(population.begin(), population.end());

    //selecciona población intermedia que se cruza
    Chromosome sel1, sel2;
    for (int i = 0; i < chr_to_cross; i+=2){
      sel1 = selection(population);
      sel2 = selection(population);
      descendents = blx_cross(sel1, sel2);

      inter_population[i]=descendents.first;
      inter_population[i+1]=descendents.second;
    }
    //selecciona la población intermedia que no se cruza
    for (int i = chr_to_cross; i < pop_size ; i++){
      inter_population[i] = selection(population);
    }

    // hace exp_num_muts mutaciones en la población intermedia
    mutate_pop(inter_population, exp_num_muts);

    // se reemplaza la población
    population = inter_population;

    // se evalua f.obj para los cruzados y mutados que no tienen valor de f.obj
    for (Chromosome& c : population){
      if (c.obj == -1){
        obj_to_chromosome(c, training);
        evals++;
      }
    }

    best_descendent = *max_element(population.begin(), population.end());

    // se mantiene el mejor de P(t) (elitismo)
    if (best_parent > best_descendent)
      *min_element(population.begin(), population.end()) = best_parent;


    if (generation % 10 == 0){
      li_ls(training, population[Random::get(0, pop_size-1)], evals);
    }

    generation++;
  }

  // se devuelven los genes del mejor cromosoma encontrado
  return max_element(population.begin(), population.end())->genes;
}

vector<double> am_1001_mej(const vector<SampleElement>& training, int cross_type){
  vector < Chromosome > population, inter_population;
  Chromosome best_parent;
  int pop_size = 10;
  int exp_num_crosses = round(0.7 * pop_size / 2.0); // núm. esperado cruces (prob cruce * población)
  int chr_to_cross = 2*exp_num_crosses; // núm. chromosomas que participan en cruce
  int exp_num_muts = 0.1 * pop_size;// número mutaciones (prob. mut. de cromosoma por total de chromosomas)
  int evals = 0;
  int generation = 1;

  pair<Chromosome, Chromosome> (*cross) (const Chromosome&, const Chromosome&);
  if (cross_type == 0)
    cross = blx_cross;
  else if (cross_type == 1)
    cross = sbx2_cross;

  population.resize(pop_size);
  // genera una población inicial aleatoria de 30 cromosomas
  for(int i = 0; i < pop_size; i++){
    for(int j = 0; j < num_feats; j++){
      population[i].genes.push_back(Random::get(0.0,1.0));
    }
    // se obtiene valor función objetivo en cada cromosoma
    obj_to_chromosome(population[i], training);
    evals++;
  }

  inter_population.resize(pop_size);
  pair<Chromosome, Chromosome> descendents;
  while (evals < 15000){
    // Se guarda el mejor cromosoma, el mejor padre
    best_parent = *max_element(population.begin(), population.end());

    //selecciona población intermedia que se cruza
    Chromosome sel1, sel2;
    for (int i = 0; i < chr_to_cross; i+=2){
      sel1 = selection(population);
      sel2 = selection(population);
      descendents = blx_cross(sel1, sel2);

      inter_population[i] = descendents.first;
      inter_population[i+1] = descendents.second;
    }
    //selecciona la población intermedia que no se cruza
    for (int i = chr_to_cross; i < pop_size ; i++){
      inter_population[i] = selection(population);
    }

    // hace exp_num_muts mutaciones en la población intermedia
    mutate_pop(inter_population, exp_num_muts);

    // se reemplaza la población
    population = inter_population;

    // se evalua f.obj para los cruzados y mutados que no tienen valor de f.obj
    for (Chromosome& c : population){
      if (c.obj == -1){
        obj_to_chromosome(c, training);
        evals++;
      }
    }

    auto best_descendent = max_element(population.begin(), population.end());

    // se mantiene el mejor de P(t) (elitismo)
    if (best_parent > *best_descendent){
      auto min_el = min_element(population.begin(), population.end());
      *min_el = best_parent;

      best_descendent = min_el;
    }



    if (generation % 10 == 0){
      li_ls(training, *best_descendent, evals);
    }

    generation++;
  }

  // se devuelven los genes del mejor cromosoma encontrado
  return max_element(population.begin(), population.end())->genes;
}

int main(int argc, char *argv[]){

	long int seed;

  if (argc <= 2) {
    cout << "Using default random seed 1234" << endl;
		seed = 1234;

  }
  else {
    seed = atoi(argv[2]);
    cout << "Using seed: " << seed << endl;
  }

	generator = default_random_engine(seed);
  Random::seed(seed);

  vector<SampleElement> sample_els;

	read_arff(argv[1], sample_els);

	normalization(sample_els);

	shuffle(sample_els.begin(), sample_els.end(), generator);

	vector<vector<SampleElement>> partitions;
  vector<SampleElement> training;
  vector<SampleElement> test;
  vector<string> classified;
  vector<double> weights(num_feats, 0.0);
  vector<double> knn_weights(num_feats, 1.0);

	cout << "--- DATASET: " << argv[1] << " ---" << endl << endl;

  clock_t start_time;
  double fin_time;
  double c_rate, r_rate;

  partitions = make_k_folds(sample_els);
  double results[7][5][4];
  string names[7] = {"AGG-BLX", "AGG-CA", "AGE-BLX", " AGE-CA", "AM-(10,1.0)", "AM-(10,0.1)", "AM-(10,0.1mej)"};

	for (int i = 0; i < K; i++){

    for (unsigned j = 0; j < i ; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());

    test = partitions[i];

    for (unsigned j = i+1; j < K; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());


		cout << "\n ---------- Partition  " << i+1 << endl;

    /* *********************************************************************** */

		cout << "AGG-BLX: \t";
    start_time = clock();

    weights = agg(training, 0);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double agg_blx_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double agg_blx_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << agg_blx_fitness << " (tiempo: " << agg_blx_time << ") " << endl;

    classified.clear();

    results[0][i][0] = c_rate;
    results[0][i][1] = r_rate;
    results[0][i][2] = agg_blx_fitness;
    results[0][i][3] = agg_blx_time;

    /* *********************************************************************** */

    cout << "AGG-CA: \t";
    start_time = clock();

    weights = agg(training, 1);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double agg_ca_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double agg_ca_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << agg_ca_fitness << " (tiempo: " << agg_ca_time << ") " << endl;

    classified.clear();

    results[1][i][0] = c_rate;
    results[1][i][1] = r_rate;
    results[1][i][2] = agg_ca_fitness;
    results[1][i][3] = agg_ca_time;

    /* *********************************************************************** */

    cout << "AGE-BLX: \t";
    start_time = clock();

    weights = age(training,0);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double age_blx_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double age_blx_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << age_blx_fitness << " (tiempo: " << age_blx_time << ") " << endl;

    classified.clear();

    results[2][i][0] = c_rate;
    results[2][i][1] = r_rate;
    results[2][i][2] = age_blx_fitness;
    results[2][i][3] = age_blx_time;

    /* *********************************************************************** */

    cout << "AGE-CA: \t";
    start_time = clock();

    weights = age(training, 1);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double age_ca_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double age_ca_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << age_ca_fitness << " (tiempo: " << age_ca_time << ") " << endl;

    classified.clear();

    results[3][i][0] = c_rate;
    results[3][i][1] = r_rate;
    results[3][i][2] = age_ca_fitness;
    results[3][i][3] = age_ca_time;

    /* *********************************************************************** */

    cout << "AM-(10,1.0): \t";
    start_time = clock();

    weights = am_1010(training,0);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double am_1010_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double am_1010_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << am_1010_fitness << " (tiempo: " << am_1010_time << ") " << endl;

    classified.clear();

    results[4][i][0] = c_rate;
    results[4][i][1] = r_rate;
    results[4][i][2] = am_1010_fitness;
    results[4][i][3] = am_1010_time;

    /* *********************************************************************** */

    cout << "AM-(10,0.1): \t";
    start_time = clock();

    weights = am_1001(training,0);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double am_1001_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double am_1001_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << am_1001_fitness << " (tiempo: " << am_1001_time << ") " << endl;

    classified.clear();

    results[5][i][0] = c_rate;
    results[5][i][1] = r_rate;
    results[5][i][2] = am_1001_fitness;
    results[5][i][3] = am_1001_time;

    /* *********************************************************************** */

    cout << "AM-(10,0.1mej): \t";
    start_time = clock();

    weights = am_1001_mej(training,0);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double am_1001_mej_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double am_1001_mej_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << am_1001_mej_fitness << " (tiempo: " << am_1001_mej_time << ") " << endl;

    classified.clear();

    results[6][i][0] = c_rate;
    results[6][i][1] = r_rate;
    results[6][i][2] = am_1001_mej_fitness;
    results[6][i][3] = am_1001_mej_time;


    fill(weights.begin(), weights.end(), 0);
    fill(knn_weights.begin(), knn_weights.end(), 0);
    training.clear();
    test.clear();
	}

  cout << "\nResultados en tabla latex:" << endl;
  for (int k = 0; k < 7; k++){
    cout << "\n" << names[k] << endl;
    for (int i=0; i < 5; i++){
      cout << results[k][i][0] << " & " << results[k][i][1] << " & " << \
      results[k][i][2] << " & " << results[k][i][3] << " \\\\ \\hline" << endl;
    }
  }

  /* Experimento adicional */

  generator = default_random_engine(seed);
  Random::seed(seed);

  double results_exp[5][5][4];
  string names_exp[5] = {"AGG-SBX2", "AGE-SBX2", "AM-(10,1.0)-SBX2", "AM-(10,0.1)-SBX2", "AM-(10,0.1mej)-SBX2"};

  cout << " Experimento adicional. " << endl;
  for (int i = 0; i < K; i++){

    for (unsigned j = 0; j < i ; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());

    test = partitions[i];

    for (unsigned j = i+1; j < K; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());


		cout << "\n ---------- Partition  " << i+1 << endl;

    /* *********************************************************************** */

		cout << "AGG-SBX2: \t";
    start_time = clock();

    weights = agg(training, 2);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double agg_sbx2_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double agg_sbx2_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << agg_sbx2_fitness << " (tiempo: " << agg_sbx2_time << ") " << endl;

    classified.clear();

    results_exp[0][i][0] = c_rate;
    results_exp[0][i][1] = r_rate;
    results_exp[0][i][2] = agg_sbx2_fitness;
    results_exp[0][i][3] = agg_sbx2_time;

    /* *********************************************************************** */

    cout << "AGE-SBX2: \t";
    start_time = clock();

    weights = age(training,2);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double age_sbx2_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double age_sbx2_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << age_sbx2_fitness << " (tiempo: " << age_sbx2_time << ") " << endl;

    classified.clear();

    results_exp[1][i][0] = c_rate;
    results_exp[1][i][1] = r_rate;
    results_exp[1][i][2] = age_sbx2_fitness;
    results_exp[1][i][3] = age_sbx2_time;

    /* *********************************************************************** */

    cout << "AM-(10,1.0)-SBX2: \t";
    start_time = clock();

    weights = am_1010(training,1);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double am_1010_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double am_1010_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << am_1010_fitness << " (tiempo: " << am_1010_time << ") " << endl;

    classified.clear();

    results_exp[2][i][0] = c_rate;
    results_exp[2][i][1] = r_rate;
    results_exp[2][i][2] = am_1010_fitness;
    results_exp[2][i][3] = am_1010_time;

    /* *********************************************************************** */

    cout << "AM-(10,0.1)-SBX2: \t";
    start_time = clock();

    weights = am_1001(training,1);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double am_1001_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double am_1001_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << am_1001_fitness << " (tiempo: " << am_1001_time << ") " << endl;

    classified.clear();

    results_exp[3][i][0] = c_rate;
    results_exp[3][i][1] = r_rate;
    results_exp[3][i][2] = am_1001_fitness;
    results_exp[3][i][3] = am_1001_time;

    /* *********************************************************************** */

    cout << "AM-(10,0.1mej)-SBX2: \t";
    start_time = clock();

    weights = am_1001_mej(training, 1);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double am_1001_mej_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double am_1001_mej_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << am_1001_mej_fitness << " (tiempo: " << am_1001_mej_time << ") " << endl;

    classified.clear();

    results_exp[4][i][0] = c_rate;
    results_exp[4][i][1] = r_rate;
    results_exp[4][i][2] = am_1001_mej_fitness;
    results_exp[4][i][3] = am_1001_mej_time;

    fill(weights.begin(), weights.end(), 0);
    fill(knn_weights.begin(), knn_weights.end(), 0);
    training.clear();
    test.clear();
	}

  cout << "\nResultados en tabla latex (exp):" << endl;
  for (int k = 0; k < 5; k++){
    cout << "\n" << names_exp[k] << endl;
    for (int i=0; i < 5; i++){
      cout << results_exp[k][i][0] << " & " << results_exp[k][i][1] << " & " << \
      results_exp[k][i][2] << " & " << results_exp[k][i][3] << " \\\\ \\hline" << endl;
    }
  }
}
