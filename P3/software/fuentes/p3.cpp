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

using namespace std;

default_random_engine generator;
int num_feats;
const int K = 5;
const double alpha = 0.5;

// Elemento muestral
struct SampleElement {
  vector<double> features;
  string label;
};

// Solución; weight y valor de f. objetivo
struct Solution{
  vector<double> w;
  double obj;
};

bool operator <(const Solution& c1, const Solution& c2){
  return c1.obj < c2.obj;
}
bool operator >(const Solution& c1, const Solution& c2){
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
    Función objetivo
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
  Asignar función objetivo a solución
*/
void obj_to_solution(Solution& s, const vector<SampleElement>& training){
  vector<string> class_labels;

  for (unsigned i = 0; i < training.size(); i++)
    class_labels.push_back( one_NN_lo(training[i], training, s.w, i) );

  s.obj = obj_function(class_rate(class_labels, training), red_rate(s.w));
}

/*
  ES. Enfriamiento simulado.
*/
vector<double> sim_annealing(const vector<SampleElement>& training){
  int n = num_feats;
  int max_neighbours = 10 * n;
  int max_success = 0.1 * max_neighbours;
  int max_evals = 15000;
  int max_anns = max_evals / max_neighbours;
  double final_temp = 0.001;
  double init_temp;
  double mu = 0.3;
  double phi = 0.3;
  normal_distribution<double> normal_dist(0.0, 0.3);
  Solution best_sol;
  int gen_neighbours;
  int evals = 0;
  double temp;
  uniform_int_distribution<int> rand_feat_ind(0, n - 1);
  uniform_real_distribution<double> rand_real(0.0, 1.0);

  Solution sol;
  for (int i = 0; i < n; i++){
    sol.w.push_back(rand_real(generator));
  }
  obj_to_solution(sol, training);
  evals = 1;

  best_sol = sol;
  init_temp = (mu * best_sol.obj) / (-1.0 * log(phi));
  temp = init_temp;

  // la temperatura final debe ser menor que la inicial
  while (final_temp >= temp)
    final_temp = final_temp * 0.1;

  double beta = (double) (init_temp - final_temp) / (max_anns * init_temp * final_temp);

  int num_success = -1;

  while (evals < max_evals && num_success != 0){
    gen_neighbours = 0;
    num_success = 0;

    while (evals < max_evals && gen_neighbours < max_neighbours && num_success < max_success){
      Solution muted_sol = sol;
      mutation(muted_sol.w, rand_feat_ind(generator), normal_dist); // operador de vecino
      obj_to_solution(muted_sol, training);
      evals++;
      gen_neighbours++;
      // decremento será mejora en la mutación
      double decr = sol.obj - muted_sol.obj;

      if (decr == 0){ // si la mutación tiene mismo fitness
        // decr igual a la temperatura final, así se irá disminuyendo probabilidad de aceptar
        // soluciones con mismo fitness conforme vaya bajando la temperatura
        // (prob 0.082 (8.2%) cuando temperatura es temp final)
        decr = 2.5 * final_temp;
      }
      if (decr < 0 || rand_real(generator) <= exp(-1.0 * decr / temp) ){
        num_success++;
        sol = muted_sol;
        if (sol.obj > best_sol.obj)
          best_sol = sol;
      }
    }
    // baja la temperatura
    temp = temp / (1.0 + beta * temp);
  }

  return best_sol.w;
}

/*
  Búsqueda local
*/
void local_search(const vector<SampleElement>& training, Solution& sol) {
  normal_distribution<double> n_dist(0.0, 0.3);
  vector<int> comp_indexes;
  int n = num_feats;
  double best_obj;
  int gen_neighbours = 0;
  int obj_eval_count = 0;
  bool new_best_obj = false;
  int mod_pos = 0;
  int max_objf_evals = 1000;
  int max_gen_neighbours = 20 * n;

  // se inicializan índices de las componentes
  for (int i = 0; i < n; i++)
    comp_indexes.push_back(i);

  best_obj = sol.obj;

  Solution muted_sol;
  int comp_to_mut;

  while ((gen_neighbours < max_gen_neighbours)  && (obj_eval_count < max_objf_evals)) {
    // aleatorizamos componentes a mutar si se han recorrido todas (o en primera iteración)
    //o se encontró nuevo mejor obj
    if (new_best_obj || (mod_pos % n == 0)){
      new_best_obj = false;
      shuffle(comp_indexes.begin(), comp_indexes.end(), generator);
      mod_pos = 0;
    }
    comp_to_mut = comp_indexes[mod_pos % n]; // componente a mutar

    muted_sol = sol;
    mutation(muted_sol.w, comp_to_mut, n_dist); // mutación
    gen_neighbours++;

    // se asigna f. obj a solución con los pesos mutados (con leave one out)
    obj_to_solution(muted_sol, training);

    // si se ha mejorado, actualizamos mejor objetivo, solución y vecinos generados
    if (muted_sol.obj > best_obj) {
      sol = muted_sol;
      best_obj = sol.obj;
      gen_neighbours = 0;
      new_best_obj = true;
    }

    obj_eval_count++;
    mod_pos++;
  }

}

/*
  BMB. Búsqueda multiarranque básica
*/
vector<double> bmb(const vector<SampleElement>& training){
  int n = num_feats;
  int max_its = 15;
  uniform_real_distribution<double> rand_real(0.0, 1.0);
  Solution sol, best_sol;

  best_sol.obj = -1;
  sol.w.resize(num_feats);

  for (int i = 0; i < max_its; i++){
    // solución aleatoria
    for (int j = 0; j < n; j++){
      sol.w[j] = rand_real(generator);
    }
    obj_to_solution(sol, training);

    local_search(training, sol);

    if (sol.obj > best_sol.obj)
      best_sol = sol;
  }

  return best_sol.w;
}

/*
  ILS. Búsqueda local iterativa
*/
vector<double> ils(const vector<SampleElement>& training){
  int n = num_feats;
  int max_its = 15;
  int t = round(0.1 * n);
  double sigma = 0.4;
  normal_distribution<double> n_dist(0.0, 0.4);
  uniform_real_distribution<double> rand_real(0.0, 1.0);

  // Solución inicial
  Solution sol;
  for (int i = 0; i < n; i++){
    sol.w.push_back(rand_real(generator));
  }
  obj_to_solution(sol, training);

  // índices a mutar
  vector<int> indexes;
  for (int i = 0; i < n; i++){
    indexes.push_back(i);
  }

  local_search(training, sol);

  for (int i = 1; i < max_its; i++){
    shuffle(indexes.begin(), indexes.end(), generator);
    Solution muted_sol = sol;

    for (int j=0; j < t; j++)
      mutation(muted_sol.w, indexes[j], n_dist);

    obj_to_solution(muted_sol, training);
    local_search(training, muted_sol);

    if (muted_sol.obj > sol.obj)
      sol = muted_sol;
  }

  return sol.w;
}

/*
  Enfriamiento simulado a partir de solución inicial
*/
Solution sim_annealing_s(const vector<SampleElement>& training, const Solution &s){
  int n = num_feats;
  int max_neighbours = 10 * n;
  int max_success = 0.1 * max_neighbours; // máximo de vecinos a aceptar
  int max_evals = 1000;
  int max_anns = max_evals / max_neighbours;
  double final_temp = 0.001;
  double init_temp;
  double mu = 0.3;
  double phi = 0.3;
  normal_distribution<double> normal_dist(0.0, 0.3);
  Solution best_sol;
  int gen_neighbours;
  int evals = 0;
  double temp;
  uniform_int_distribution<int> rand_feat_ind(0, n - 1);
  uniform_real_distribution<double> rand_real(0.0, 1.0);

  Solution sol = s;

  best_sol = sol;
  init_temp = (mu * best_sol.obj) / (-1.0 * log(phi));
  temp = init_temp;

  // la temperatura final debe ser menor que la inicial
  while (final_temp >= temp)
    final_temp = final_temp * 0.1;

  double beta = (double) (init_temp - final_temp) / (max_anns * init_temp * final_temp);

  int num_success = -1;

  while (evals < max_evals && num_success != 0){
    gen_neighbours = 0;
    num_success = 0;

    while (evals < max_evals && gen_neighbours < max_neighbours && num_success < max_success){
      Solution muted_sol = sol;
      mutation(muted_sol.w, rand_feat_ind(generator), normal_dist); // operador de vecino
      obj_to_solution(muted_sol, training);
      evals++;
      gen_neighbours++;
      // decremento será mejora en la mutación
      double decr = sol.obj - muted_sol.obj;

      if (decr == 0){ // si la mutación tiene mismo fitness
        // decr igual a la temperatura final, así se irá disminuyendo probabilidad de aceptar
        // soluciones con mismo fitness conforme vaya bajando la temperatura
        // (prob 0.082 (8.2%) cuando temperatura es temp final)
        decr = 2.5 * final_temp;
      }
      // si se ha mejorado o por cierto factor aleatorio, se acepta la mutación y contabiliza éxito
      if (decr < 0 || rand_real(generator) <= exp(-1.0 * decr / temp) ){
        num_success++;
        sol = muted_sol;
        if (sol.obj > best_sol.obj)
          best_sol = sol;
      }
    }
    // baja la temperatura
    temp = temp / (1.0 + beta * temp);
  }

  return best_sol;
}

/*
  ILS-ES
*/
vector<double> ils_es(const vector<SampleElement>& training){
  int max_its = 15;
  int t = round(0.1 * num_feats);
  double sigma = 0.4;
  normal_distribution<double> n_dist(0.0, 0.4);
  uniform_int_distribution<int> rand_feat_ind(0, num_feats-1);
  uniform_real_distribution<double> rand_real(0.0, 1.0);

  // Solución inicial
  Solution sol;
  for (int i = 0; i < num_feats; i++){
    sol.w.push_back(rand_real(generator));
  }
  obj_to_solution(sol, training);

  // índices a mutar
  vector<int> indexes;
  for (int i = 0; i < num_feats; i++){
    indexes.push_back(i);
  }

  sol = sim_annealing_s(training, sol);

  for (int i = 1; i < max_its; i++){
    shuffle(indexes.begin(), indexes.end(), generator);
    Solution muted_sol = sol;

    for (int j = 0; j < t; j++)
      mutation(muted_sol.w, indexes[i], n_dist);

    obj_to_solution(muted_sol, training);
    muted_sol = sim_annealing_s(training, muted_sol);

    if (muted_sol.obj > sol.obj)
      sol = muted_sol;
  }

  return sol.w;
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
  double results[4][5][4];
  string names[7] = {"ES", "BMB", "ILS", "ILS_ES"};

	for (int i = 0; i < K; i++){

    for (unsigned j = 0; j < i ; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());

    test = partitions[i];

    for (unsigned j = i+1; j < K; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());


		cout << "\n ---------- Partition  " << i+1 << endl;

    /* *********************************************************************** */

		cout << "ES: \t";
    start_time = clock();

    weights = sim_annealing(training);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double es_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double es_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << es_fitness << " (tiempo: " << es_time << ") " << endl;

    classified.clear();

    results[0][i][0] = c_rate;
    results[0][i][1] = r_rate;
    results[0][i][2] = es_fitness;
    results[0][i][3] = es_time;

    /* *********************************************************************** */

		cout << "BMB: \t";
    start_time = clock();

    weights = bmb(training);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double bmb_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double bmb_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << bmb_fitness << " (tiempo: " << bmb_time << ") " << endl;

    classified.clear();

    results[1][i][0] = c_rate;
    results[1][i][1] = r_rate;
    results[1][i][2] = bmb_fitness;
    results[1][i][3] = bmb_time;

    /* *********************************************************************** */

    cout << "ILS: \t";
    start_time = clock();

    weights = ils(training);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double ils_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double ils_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << ils_fitness << " (tiempo: " << ils_time << ") " << endl;

    classified.clear();

    results[2][i][0] = c_rate;
    results[2][i][1] = r_rate;
    results[2][i][2] = ils_fitness;
    results[2][i][3] = ils_time;

    /* *********************************************************************** */

    cout << "ILS-ES: \t";
    start_time = clock();

    weights = ils_es(training);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double ils_es_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double ils_es_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << ils_es_fitness << " (tiempo: " << ils_es_time << ") " << endl;

    classified.clear();

    results[3][i][0] = c_rate;
    results[3][i][1] = r_rate;
    results[3][i][2] = ils_es_fitness;
    results[3][i][3] = ils_es_time;




    fill(weights.begin(), weights.end(), 0);
    fill(knn_weights.begin(), knn_weights.end(), 0);
    training.clear();
    test.clear();
	}


  cout << "\nResultados en tabla latex:" << endl;
  double mean[4];
  for (int k = 0; k < 4; k++){
    cout << "\n" << names[k] << endl;
    for (int i=0; i < 5; i++){
      cout << results[k][i][0] << " & " << results[k][i][1] << " & " << \
      results[k][i][2] << " & " << results[k][i][3] << " \\\\ \\hline" << endl;
    }

    for (int j = 0; j < 4; j++)
      mean[j] = 0;

    for (int i = 0; i < 5; i++)
      for (int j = 0; j < 4; j++)
        mean[j] += results[k][i][j];

    cout << mean[0]/5.0 << " & " << mean[1]/5.0 << " & " << \
    mean[2]/5.0 << " & " << mean[3]/5.0 << " \\\\ \\hline" << endl;
  }

}
