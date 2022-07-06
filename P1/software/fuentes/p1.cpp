#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <limits>
#include <cassert>

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

		// ALMACENAMOS LAS VARIABLES GLOBALES
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
  LOCAL SEARCH
*/

void rand_init_sol_gen_indexes(vector<double>& weights, vector<int>& indexes){
  uniform_real_distribution<double> distribution(0.0, 1.0);

  for (int i=0; i < num_feats; i++){
    weights.push_back(distribution(generator));
    indexes.push_back(i);
  }
}

void mutation(vector<double>& weights, int i, normal_distribution<double>& n_dist){
	weights[i] += n_dist(generator);

	if (weights[i] > 1)
    weights[i] = 1;
	else {
    if (weights[i] < 0)
      weights[i] = 0;
  }
}

double classifyloo_and_objf(const vector<SampleElement>& training, const vector<double>& weights, vector<string>& class_labels){
  // clasificamos los elementos, con leave one out
  for (unsigned k = 0; k < training.size(); k++)
    class_labels.push_back(one_NN_lo(training[k], training, weights, k));
  // obtenemos valor de función objetivo
  double obj = obj_function(class_rate(class_labels, training), red_rate(weights));
  class_labels.clear();

  return obj;
}

vector<double> local_search(const vector<SampleElement>& training){
  normal_distribution<double> n_dist(0.0, 0.3);
  vector<double> best_weights;
  int max_gen_neighbours = 20 * num_feats;
  int max_objf_evals = 15000;
  int obj_eval_count = 0;
  int gen_neighbours = 0;

  vector<string> class_labels;
  bool new_best_obj = false;

  // genera pesos iniciales e inicializamos índices a mutar
  vector<int> comp_indexes;
  rand_init_sol_gen_indexes(best_weights, comp_indexes);

  // clasificamos con los mejores pesos hasta el momento mediante leave one out,
  // y obtenemos valor de la función objetivo
  double best_obj = classifyloo_and_objf(training, best_weights, class_labels);

  vector<double> muted_weights;
  double obj;
  int comp_to_mut;
  int mod_pos = 0;

  while ( (gen_neighbours < max_gen_neighbours) && (obj_eval_count < max_objf_evals) ){
    // aleatorizamos componentes a mutar si se han recorrido todas o se mejoró f.obj
    if (new_best_obj || (mod_pos % num_feats == 0)){
      new_best_obj = false;
      shuffle(comp_indexes.begin(), comp_indexes.end(), generator);
      mod_pos = 0;
    }

    comp_to_mut = comp_indexes[mod_pos % num_feats]; // componente a mutar

    muted_weights = best_weights;

    mutation(muted_weights, comp_to_mut, n_dist); // mutación
    gen_neighbours++; //incrementamos vecinos generados

    // clasificamos con los nuevos pesos mutados mediante leave one out, y obtenemos valor de la función objetivo
    obj = classifyloo_and_objf(training, muted_weights, class_labels);

    // si se ha mejorado, actualizamos mejor objetivo y weights, y vecinos generados
    if (obj > best_obj){
      best_weights = muted_weights;
      best_obj = obj;
      gen_neighbours = 0;

      new_best_obj = true;
    }

    obj_eval_count++;
    mod_pos++;
  }

	return best_weights;
}


/*
    RELIEF
*/

void closest_friend_enemy(unsigned pos, const vector<SampleElement>& training, SampleElement& friend_e, SampleElement& enemy_e){
  double min_friend_d = numeric_limits<double>::max();
  double min_enemy_d = numeric_limits<double>::max();
  unsigned min_friend_p;
  unsigned min_enemy_p;
  double distance;
  SampleElement e;

  for (unsigned i = 0; i < pos; i++){
      distance = euclidean_distance2(training[i].features, training[pos].features);
      // si es amigo
      if (training[pos].label == training[i].label){
        if (distance < min_friend_d){
          min_friend_d = distance;
          min_friend_p = i;
        }
      }
      else{ // si es enemigo
        if(distance < min_enemy_d){
          min_enemy_d = distance;
          min_enemy_p = i;
        }
      }
  }

  for (unsigned i = pos+1; i < training.size(); i++){
      distance = euclidean_distance2(training[i].features, training[pos].features);
      // si es amigo
      if (training[pos].label == training[i].label){
        if (distance < min_friend_d){
          min_friend_d = distance;
          min_friend_p = i;
        }
      }
      else{ // si es enemigo
        if(distance < min_enemy_d){
          min_enemy_d = distance;
          min_enemy_p = i;
        }
      }
  }

  friend_e = training[min_friend_p];
  enemy_e = training[min_enemy_p];
}

vector<double> relief(const vector<SampleElement>& training){
	vector<double> weights(num_feats, 0.0);
  SampleElement friend_e, enemy_e;

	for (int i = 0; i < training.size(); i++){
    closest_friend_enemy(i, training, friend_e, enemy_e);

		for (int j = 0; j < num_feats; j++){
			weights[j] += abs(training[i].features[j]-enemy_e.features[j]) \
        - abs(training[i].features[j]-friend_e.features[j]);
		}
	}

	double max_weight = *max_element(weights.begin(), weights.end());

	for (int j=0; j < num_feats; j++){
		if (weights[j] < 0)
			weights[j] = 0;
		else
			weights[j] = weights[j] / max_weight;
	}

	return weights;
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

	for (int i = 0; i < K; i++){

    for (unsigned j = 0; j < i ; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());

    test = partitions[i];

    for (unsigned j = i+1; j < K; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());


		cout << "\n ---------- Partition  " << i+1 << endl;

    /* *********************************************************************** */

		cout << "1-NN: \t";

		start_time = clock();

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, knn_weights) );

    fin_time = clock();
		double knn_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    double knn_fitness = obj_function(c_rate, 0.0);

    cout << "\t Tasa clasificación: " << c_rate << "\t Tasa reducción: " << 0.0 << endl;
		cout << "\t\t Función objetivo: " << knn_fitness << " (tiempo: " << knn_time << ") " << endl;

    classified.clear();

    /* *********************************************************************** */

		cout << "Local Search: \t";

    start_time = clock();

    weights = local_search(training);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
		double ls_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double ls_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
		cout << "\t\t Función objetivo: " << ls_fitness << " (tiempo: " << ls_time << ") " << endl;

    classified.clear();

    /* *********************************************************************** */

		cout << "RELIEF: \t";
    start_time = clock();

    weights = relief(training);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double relief_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double relief_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << relief_fitness << " (tiempo: " << relief_time << ") " << endl;

    classified.clear();


    fill(weights.begin(), weights.end(), 0);
    fill(knn_weights.begin(), knn_weights.end(), 0);
    training.clear();
    test.clear();
	}

}
