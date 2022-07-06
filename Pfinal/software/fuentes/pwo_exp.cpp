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

// Solución; posición y valor de f. objetivo
struct Wolf{
  vector<double> pos;
  double obj;
};

bool operator <(const Wolf& c1, const Wolf& c2){
  return c1.obj < c2.obj;
}
bool operator >(const Wolf& c1, const Wolf& c2){
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
  Asignar función objetivo a wolf
*/
void obj_to_wolf(Wolf& w, const vector<SampleElement>& training){
  vector<string> class_labels;

  for (unsigned i = 0; i < training.size(); i++)
    class_labels.push_back( one_NN_lo(training[i], training, w.pos, i) );

  w.obj = obj_function(class_rate(class_labels, training), red_rate(w.pos));
}

/* Restringir posiciones (pesos) a [0,1] */
void restrict_01(Wolf& w){
  for (int i=0; i < num_feats; i++){
    if (w.pos[i] < 0.0)
      w.pos[i] = 0.0;
    else if (w.pos[i] > 1.0)
      w.pos[i] = 1.0;
  }
}

/* Inicializar posiciones de lobos */
void init_wolf_pos(Wolf& w, uniform_real_distribution<double>& rand_real_01){
  w.pos.resize(num_feats);
  for (int i = 0; i < num_feats; i++){
    w.pos[i] = rand_real_01(generator);
  }
}
/* Actualizar posiciones y valores de función objetivo */
void update_alpha_beta_delta(const vector<Wolf>& wolfs, Wolf& alpha, Wolf& beta, Wolf& delta){
  for (auto& w : wolfs){
    if (w.obj > alpha.obj){
      alpha.obj = w.obj;
      alpha.pos = w.pos;
    }
    else{
      if (w.obj > beta.obj && w.obj < alpha.obj){
        beta.obj = w.obj;
        beta.pos = w.pos;
      }
      else if (w.obj > delta.obj && w.obj < beta.obj && w.obj < alpha.obj){
        delta.obj = w.obj;
        delta.pos = w.pos;
      }
    }
  }
}

/*
  GWO. Grey Wolf Optimizer
*/
vector<double> gwo(const vector<SampleElement>& training, int num_agents, int max_evals){
  int n = num_feats;
  uniform_real_distribution<double> rand_real_01(0.0, 1.0);
  int max_iters = max_evals / num_agents;

  vector<Wolf> wolfs;
  // se inicializan posiciones y asignan valores de f. objetivo para cada agente
  for (int i = 0; i < num_agents; i++){
    Wolf w;
    init_wolf_pos(w, rand_real_01);
    obj_to_wolf(w, training);
    wolfs.push_back(w);
  }
  int evals = num_agents;

  sort(wolfs.begin(), wolfs.end()); // ordena de menor a mayor f.objetivo

  // guardar la posición y valor de f.objetivo de alpha, beta y delta
  Wolf alpha = wolfs[num_agents-1];
  Wolf beta = wolfs[num_agents-2];
  Wolf delta = wolfs[num_agents-3];

  int it = 1;
  while (evals < max_evals){
    // actualiza a que decrece de 2 a 0
    double a = 2 - 2.0 * ((double) it) / max_iters;

    // Actualiza las posiciones de cada agente incluyendo omegas
    for (auto w(wolfs.begin()); w != wolfs.end() && evals < max_evals; ++w){
      for (int j = 0; j < n; j++){
        double r1 = rand_real_01(generator);
        double r2 = rand_real_01(generator);

        double A1 = 2.0 * a * r1 - a;
        double C1 = 2.0 * r2;

        double D_alpha = abs(C1 * alpha.pos[j] - w->pos[j]);
        double X1 = alpha.pos[j] - A1 * D_alpha;

        r1 = rand_real_01(generator);
        r2 = rand_real_01(generator);

        double A2 = 2.0 * a * r1 - a;
        double C2 = 2.0 * r2;

        double D_beta = abs(C2 * beta.pos[j] - w->pos[j]);
        double X2 = beta.pos[j] - A2 * D_beta;

        r1 = rand_real_01(generator);
        r2 = rand_real_01(generator);

        double A3 = 2.0 * a * r1 - a;
        double C3 = 2.0 * r2;

        double D_delta = abs(C3 * delta.pos[j] - w->pos[j]);
        double X3 = delta.pos[j] - A3 * D_delta;

        w->pos[j] = (X1 + X2 + X3) / 3.0;
      }
      // restringe a [0,1] y asigna función objetivo
      restrict_01(*w);
      obj_to_wolf(*w, training);
      evals++;
    }

    update_alpha_beta_delta(wolfs, alpha, beta, delta);
    it++;
  }

  return alpha.pos;
}

/*
  low intensity local search
*/
void li_ls(const vector<SampleElement>& training, Wolf& c, int& obj_eval_count) {
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

  Wolf muted_c;
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
    mutation(muted_c.pos, comp_to_mut, n_dist); // mutación
    gen_neighbours++;

    // se asigna f. obj a cromosoma con los pesos mutados (con leave one out)
    obj_to_wolf(muted_c, training);

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

/*
  GWO. Grey Wolf Optimizer
*/
vector<double> gwo_ls(const vector<SampleElement>& training, int num_agents, int max_evals){
  int n = num_feats;
  uniform_real_distribution<double> rand_real_01(0.0, 1.0);
  int max_iters = max_evals / num_agents;

  vector<Wolf> wolfs;
  // se inicializan posiciones y asignan valores de f. objetivo para cada agente
  for (int i = 0; i < num_agents; i++){
    Wolf w;
    init_wolf_pos(w, rand_real_01);
    obj_to_wolf(w, training);
    wolfs.push_back(w);
  }
  int evals = num_agents;

  sort(wolfs.begin(), wolfs.end()); // ordena de menor a mayor f.objetivo

  // guardar la posición y valor de f.objetivo de alpha, beta y delta
  Wolf alpha = wolfs[num_agents-1];
  Wolf beta = wolfs[num_agents-2];
  Wolf delta = wolfs[num_agents-3];

  int it = 1;
  while (evals < max_evals){
    // actualiza a que decrece de 2 a 0
    double a = 2 - 2.0 * ((double) it) / max_iters;

    // Actualiza las posiciones de cada agente incluyendo omegas
    for (auto w(wolfs.begin()); w != wolfs.end() && evals < max_evals; ++w){
      for (int j = 0; j < n; j++){
        double r1 = rand_real_01(generator);
        double r2 = rand_real_01(generator);

        double A1 = 2.0 * a * r1 - a;
        double C1 = 2.0 * r2;

        double D_alpha = abs(C1 * alpha.pos[j] - w->pos[j]);
        double X1 = alpha.pos[j] - A1 * D_alpha;

        r1 = rand_real_01(generator);
        r2 = rand_real_01(generator);

        double A2 = 2.0 * a * r1 - a;
        double C2 = 2.0 * r2;

        double D_beta = abs(C2 * beta.pos[j] - w->pos[j]);
        double X2 = beta.pos[j] - A2 * D_beta;

        r1 = rand_real_01(generator);
        r2 = rand_real_01(generator);

        double A3 = 2.0 * a * r1 - a;
        double C3 = 2.0 * r2;

        double D_delta = abs(C3 * delta.pos[j] - w->pos[j]);
        double X3 = delta.pos[j] - A3 * D_delta;

        w->pos[j] = (X1 + X2 + X3) / 3.0;
      }
      // restringe a [0,1] y asigna función objetivo
      restrict_01(*w);
      obj_to_wolf(*w, training);
      evals++;
    }

    update_alpha_beta_delta(wolfs, alpha, beta, delta);

    if (it % 30 == 0){
      li_ls(training, alpha, evals);
      li_ls(training, beta, evals);
      li_ls(training, delta, evals);
    }

    it++;

  }

  return alpha.pos;
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
  string names[7] = {"GWO20", "GWO20_LS", "GWO30", "GWO30_LS"};

	for (int i = 0; i < K; i++){

    for (unsigned j = 0; j < i ; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());

    test = partitions[i];

    for (unsigned j = i+1; j < K; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());


		cout << "\n ---------- Partition  " << i+1 << endl;

    /* *********************************************************************** */

		cout << "GWO: \t";
    start_time = clock();

    weights = gwo(training, 20, 15000);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double gwo_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double gwo_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << gwo_fitness << " (tiempo: " << gwo_time << ") " << endl;

    classified.clear();

    results[0][i][0] = c_rate;
    results[0][i][1] = r_rate;
    results[0][i][2] = gwo_fitness;
    results[0][i][3] = gwo_time;


    /* *********************************************************************** */

    cout << "GWO_LS: \t";
    start_time = clock();

    weights = gwo_ls(training, 20, 15000);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double gwo_ls_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double gwo_ls_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << gwo_ls_fitness << " (tiempo: " << gwo_ls_time << ") " << endl;

    classified.clear();

    results[1][i][0] = c_rate;
    results[1][i][1] = r_rate;
    results[1][i][2] = gwo_ls_fitness;
    results[1][i][3] = gwo_ls_time;

    /* *********************************************************************** */

		cout << "GWO: \t";
    start_time = clock();

    weights = gwo(training, 30, 15000);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double gwo30_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double gwo30_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << gwo30_fitness << " (tiempo: " << gwo30_time << ") " << endl;

    classified.clear();

    results[2][i][0] = c_rate;
    results[2][i][1] = r_rate;
    results[2][i][2] = gwo_fitness;
    results[2][i][3] = gwo_time;


    /* *********************************************************************** */

    cout << "GWO_LS: \t";
    start_time = clock();

    weights = gwo_ls(training, 30, 15000);

    for (int k = 0; k < test.size(); k++)
      classified.push_back( one_NN(test[k], training, weights) );

    fin_time = clock();
    double gwo30_ls_time = 1000.0 * (fin_time - start_time)/CLOCKS_PER_SEC;

    c_rate = class_rate(classified, test);
    r_rate = red_rate(weights);
    double gwo30_ls_fitness = obj_function(c_rate, r_rate);

    cout << " Tasa clasificación: " << c_rate << "\t Tasa reducción: " << r_rate << endl;
    cout << "\t\t Función objetivo: " << gwo_ls_fitness << " (tiempo: " << gwo_ls_time << ") " << endl;

    classified.clear();

    results[3][i][0] = c_rate;
    results[3][i][1] = r_rate;
    results[3][i][2] = gwo_ls_fitness;
    results[3][i][3] = gwo_ls_time;


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
