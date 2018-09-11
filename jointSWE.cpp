#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <tuple>
#include <unordered_map>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <map>
#include <cmath>
#include <ctgmath>
#include <cstring>
#include <functional>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <stdio.h>

#include <boost/algorithm/string.hpp>
#include <Eigen/Dense>
#include "omp.h"

#include <pthread.h>
#include <boost/optional.hpp>
#include "parse_args.hh"

using namespace std;
using namespace boost;
using namespace Eigen;
using Eigen::VectorXd;


float X_ONE   = 1;
float X_TWO   = 1;
float X_THREE = 1;
struct edge {
    size_t main;
    size_t context;
    size_t pair_type;
    size_t main_tag;
    size_t context_tag;
    size_t context_word_id;
};

struct TaggedWord {
    string Word;
    string Tag;
    bool IsTagged;
    size_t TagId;
    size_t WordId;
    TaggedWord(const string &text) {
        if (text.find('/') == string::npos ) {
            Word = text;
            IsTagged = false;
            return;
        }
        size_t tag_start = text.find('/');
        Word = text.substr(0, tag_start);
        Tag = text.substr(tag_start + 1, text.size() - tag_start - 1);
        IsTagged = true;
    }
};


struct PairsContent {
    vector<size_t> UnigramTable;
    vector<edge> WordPairs;
    vector<size_t> WordPairsIndex;
};

set<size_t> words;
set<size_t> contexts;
set<size_t> tags_words;
set<size_t> tags_contexts;

unordered_map<string, size_t> word_to_index_map;
unordered_map<string, size_t> tag_to_index_map;

vector<TaggedWord> strings;
vector<string> tag_names;

int D; // dimensionality
int sample_count;
int epohs;
float alpha;
int no_threads;

unordered_map<size_t, VectorXd> w; // target word vectors
unordered_map<size_t, VectorXd> c; // context word vectors
unordered_map<size_t, VectorXd> w_t; // main tag vectors
unordered_map<size_t, VectorXd> c_t; // context tag vectors

unordered_map<size_t, VectorXd> grad_w; // Squared gradient for AdaGrad
unordered_map<size_t, VectorXd> grad_c;
unordered_map<size_t, VectorXd> grad_w_t;
unordered_map<size_t, VectorXd> grad_c_t;



size_t get_tag_id(const string &str, bool fill_strings = false) {
    if (tag_to_index_map.find(str) == tag_to_index_map.end()) {
        size_t id = tag_to_index_map.size();
        tag_to_index_map[str] = id;
        if (fill_strings) {
            tag_names.push_back(str);
        }
    }
    return tag_to_index_map[str];
}
size_t get_string_id(const string &str, bool fill_strings = false) {
    if (word_to_index_map.find(str) == word_to_index_map.end()) {
        size_t id = strings.size();
        word_to_index_map[str] = id;
        TaggedWord tagged_word(str);
        if (tagged_word.IsTagged) {
            size_t  tag_id = get_tag_id(tagged_word.Tag, true);
            size_t word_id = get_string_id(tagged_word.Word);
            tagged_word.TagId = tag_id;
            tagged_word.WordId = word_id;
        }
        strings.push_back(tagged_word);

    }
    return word_to_index_map[str];
}

void SetPairTypeAndTags(edge& e) {
    if (!(strings[e.main].IsTagged) && !(strings[e.context].IsTagged)) {
        e.pair_type = 0;
    }
    else if (strings[e.main].IsTagged && !(strings[e.context].IsTagged)) {
        e.pair_type = 2;

        e.main_tag = strings[e.main].TagId;
        tags_words.insert(e.main_tag);
        words.insert(strings[e.main].WordId);
    }
    else if (!(strings[e.main].IsTagged) && strings[e.context].IsTagged) {
        e.pair_type =  1;
        e.context_tag = strings[e.context].TagId;
        tags_contexts.insert(e.context_tag);
        e.context_word_id = strings[e.context].WordId;
        contexts.insert(e.context_word_id);
    } else {
        e.pair_type =  3;
        e.context_tag = strings[e.context].TagId;
        tags_contexts.insert(e.context_tag);
        e.context_word_id = strings[e.context].WordId;
        contexts.insert(e.context_word_id);
        e.main_tag = strings[e.main].TagId;
        tags_words.insert(e.main_tag);
        words.insert(strings[e.main].WordId);
    }

}

int read_pairs(string fname, vector<edge> &wordpairs, vector<size_t>& wordpairsindex) {
    ifstream input;
    input.open(fname);
    string m;
    string c;
    size_t count;
    int pair_counter = 0;
    cout << "Reading Pair File.. " << fname << endl;
    while (input >> m >> c >> count) {
        edge e;
        e.main = get_string_id(m, true);
        e.context = get_string_id(c, true);
        SetPairTypeAndTags(e);
        wordpairs.push_back(e);
        words.insert(e.main);
        contexts.insert(e.context);
        for (auto i = 0; i < count; ++i)
            wordpairsindex.push_back(pair_counter);
        pair_counter++;
    }
    cout << endl;
    cout << "Word Pairs Loaded..." << endl;
    input.close();
}

int load_unigramT(string fname, vector<size_t> &unigram_table) {
    cout << "Loading unigram table..." << endl;
    ifstream input;
    input.open(fname);
    string word;
    while (input >> word) {
        size_t ind = get_string_id(word, true);
        if (contexts.find(ind) != contexts.end())
            unigram_table.push_back(ind);
    }
    cout << "Completed Successfully..." << endl;

}

void centralize(unordered_map<size_t, VectorXd> &x) {
    VectorXd mean = VectorXd::Zero(D);
    VectorXd squared_mean = VectorXd::Zero(D);
    for (auto w = x.begin(); w != x.end(); ++w) {
        mean += w->second;
        squared_mean += (w->second).cwiseProduct(w->second);
    }
    mean = mean / ((double) x.size());
    VectorXd sd = squared_mean - mean.cwiseProduct(mean);
    for (int i = 0; i < D; ++i) {
        sd[i] = sqrt(sd[i]);
    }
    for (auto w = x.begin(); w != x.end(); ++w) {
        VectorXd tmp = VectorXd::Zero(D);
        for (int i = 0; i < D; ++i) {
            tmp[i] = (w->second)[i] - mean[i];
            if (sd[i] != 0)
                tmp[i] /= sd[i];
        }
        w->second = tmp;
    }
}

void initialize() {
    int count_words = 0;
    cout << "Initializing Vectors..." << endl;
    for (auto e = words.begin(); e != words.end(); ++e) {
        count_words++;
        w[*e] = VectorXd::Random(D);
        grad_w[*e] = VectorXd::Zero(D);
    }

    int count_contexts = 0;
    for (auto e = contexts.begin(); e != contexts.end(); ++e) {
        count_contexts++;
        c[*e] = VectorXd::Random(D);
        grad_c[*e] = VectorXd::Zero(D);
    }

    for (auto e = tags_words.begin(); e != tags_words.end(); ++e) {
        w_t[*e] = VectorXd::Random(D);
        grad_w_t[*e] = VectorXd::Zero(D);
    }

    for (auto e = tags_contexts.begin(); e != tags_contexts.end(); ++e) {
        c_t[*e] = VectorXd::Random(D);
        grad_c_t[*e] = VectorXd::Zero(D);
    }


    cout << "Centralizing Vectors..." << endl;
    centralize(w);
    centralize(c);
    centralize(w_t);
    centralize(c_t);

    cout << "Initialization Phase Completed..." << endl;
}


struct TrainArgs {
    size_t ThreadId;
    size_t StartPoint;
    size_t EndPoint;
    PairsContent *PairContent;
};

size_t GetNextRandom(size_t id, vector<size_t>& unigram_table) {
    size_t next_random = ((rand() % unigram_table.size()) + id) % unigram_table.size();
    return next_random;
}
void *train(void *a) {
    TrainArgs *args = (TrainArgs *) a;
    double score;
    VectorXd gw = VectorXd::Zero(D);
    VectorXd gc = VectorXd::Zero(D);
    VectorXd gcr = VectorXd::Zero(D);

    VectorXd gwt = VectorXd::Zero(D);
    VectorXd gct = VectorXd::Zero(D);
    VectorXd gcrt = VectorXd::Zero(D);

    VectorXd diff = VectorXd::Zero(D);
    VectorXd one_vect = VectorXd::Zero(D);

    for (auto i = 0; i < D; ++i)
        one_vect[i] = 1.0;

    int pair_count = 0;

    int total_pairs = args->PairContent->WordPairs.size();
    //generate_n_samples(sample_count,random_samples);
    string m, con;
    srand(time(NULL));
    int smp = 0;
    auto &word_pairs = args->PairContent->WordPairs;
    auto &word_pairs_index = args->PairContent->WordPairsIndex;
    auto &unigram_table = args->PairContent->UnigramTable;
    auto &id = args->ThreadId;
    for (int e = args->StartPoint; e < args->EndPoint; ++e) {
        pair_count++;
        for (int i = 0; i < sample_count; i++) {
            size_t next_random = GetNextRandom(id, unigram_table);

            auto &pair = word_pairs[word_pairs_index[e]];
            auto& pair_type  = pair.pair_type;
            bool to_update = false;

            size_t main_id = pair.main;
            size_t main_tag_id = -1;

            if (strings[main_id].IsTagged) {
                main_tag_id = strings[main_id].TagId;
                main_id = strings[main_id].WordId;
            }
            size_t context_id = pair.context;
            size_t context_tag_id = -1;
            if (strings[context_id].IsTagged) {
                context_tag_id = strings[context_id].TagId;
                context_id = strings[context_id].WordId;
            }

            size_t random_id = unigram_table[next_random];
            size_t random_tag_id = -1;
            VectorXd  curr_tr = c[random_id];// if the random word is not tagged take context vector instead of tags vector
            bool tagged_random = false;
            if (strings[random_id].IsTagged) {
                random_id = strings[random_id].WordId;
                random_tag_id = strings[random_id].TagId;
                if (c_t.find(random_tag_id) != c_t.end()) {
                    curr_tr = c_t[random_tag_id];
                    tagged_random = true;
                }
            }

            auto score_one = w[main_id].dot(c[context_id] - c[random_id]);
            float score_two = 1;
            if (pair.pair_type == 1 || pair.pair_type == 3) {
                score_two = X_ONE * w[main_id].dot(curr_tr - c_t[context_tag_id]);
            }
            float score_three  = 1;
            if (pair.pair_type == 2 || pair.pair_type ==3) {
                score_three = X_TWO * w_t[main_tag_id].dot(w[main_id] - c[context_id]);
            }
            float score_four = 1;
            if (pair.pair_type == 3) {
                score_four = X_THREE * w_t[main_tag_id].dot(curr_tr - c_t[context_tag_id]);
            }

             vector<bool> updates(6, false);


            if (pair_type == 0 && score_one < 1) {
                gw = (c[context_id] * -1) + c[random_id];
                gc = w[main_id] * -1;
                gcr = w[main_id];
                updates[0] = true;
                updates[1] = true;
                updates[2] = true;
            }
            if (pair_type == 1 ) {
                if (score_one < 1 && score_two < 1) {
                    gw = c[pair.context]  + (c[random_id] * -1) - ((c_t[context_tag_id] +  (curr_tr * -1 )) * X_ONE);
                    updates[0] = true;
                } else if (score_one < 1) {
                    gw = (c[context_id] * -1) + c[random_id];
                    updates[0] = true;
                } else if (score_two < 1) {
                    gw = ((c_t[context_tag_id] +  (curr_tr * -1 )) * X_ONE);
                    updates[0] = true;
                }
                if (score_one < 1) {
                    gc = w[main_id] * -1;
                    gcr = w[main_id];
                    updates[1] = true;
                    updates[2] = true;
                }
                if (score_two < 1) {
                    gct = w[main_id] * -1;
                    gcrt =  w[main_id];
                    updates[4] = true;
                    updates[5] = true;
                }
            }

            if (pair_type == 2) {
                if (score_one < 1 && score_three < 1) {
                    gc = w[main_id] + (w_t[main_tag_id] * -X_THREE);
                    gcr = (w[main_id] * -1) + (w_t[main_tag_id] * X_THREE);
                    updates[1] = true;
                    updates[2] = true;
                } else if (score_one < 1) {
                    gc = w[main_id] * -1;
                    gcr = w[main_id];
                    updates[1] = true;
                    updates[2] = true;
                } else if (score_three < 1) {
                    gc = X_TWO * w_t[main_tag_id];
                    gcr = -X_TWO * w_t[main_tag_id];
                    updates[1] = true;
                    updates[2] = true;
                }
                if (score_one < 1) {
                    gw = (c[context_id] * -1) + c[random_id];
                    updates[0] = true;
                }
                if (score_three < 1) {
                    gwt = -X_TWO * c[context_id];
                    updates[3] = true;
                }
            }

            if (pair_type == 3) {
                if (score_one < 1 && score_two < 1) {
                    gw = c[pair.context]  + (c[random_id] * -1) + ((c_t[context_tag_id] +  (curr_tr * -1 )) * X_ONE);
                    updates[0] = true;
                } else if (score_one < 1) {
                    gw = (c[context_id] * -1) + c[random_id];
                    updates[0] = true;
                } else if (score_two < 1) {
                    gw = ((c_t[context_tag_id] +  (curr_tr * -1 )) * X_ONE);
                    updates[0] = true;
                }
                if (score_one < 1 && score_three < 1) {
                    gc = w[main_id] + (w_t[main_tag_id] * -X_THREE);
                    gcr = (w[main_id] * -1) + (w_t[main_tag_id] * X_THREE);
                    updates[1] = true;
                    updates[2] = true;
                } else if (score_one < 1) {
                    gc = w[main_id] * -1;
                    gcr = w[main_id];
                    updates[1] = true;
                    updates[2] = true;
                } else if (score_three < 1) {
                    gc = X_TWO * w_t[main_tag_id];
                    gcr = -X_TWO * w_t[main_tag_id];
                    updates[1] = true;
                    updates[2] = true;
                }

                if ( score_three < 1 && score_four < 1) {
                    gwt = X_TWO * c[context_id] - X_THREE * (c_t[context_tag_id] - curr_tr);
                    updates[3] = true;
                } else if (score_three < 1) {
                    gwt = X_TWO * c[context_id];
                    updates[3] = true;
                } else if (score_four < 1) {
                    gwt = X_THREE * (c_t[context_tag_id] - curr_tr);
                    updates[3] = true;
                }

                if (score_two < 1 && score_four < 1) {
                    gct = X_ONE * w[main_id] - X_THREE * w_t[main_tag_id];
                    gcrt = -X_ONE * w[main_id] + X_THREE * w_t[main_tag_id];
                    updates[4] = true;
                    updates[5] = true;
                } else if (score_two < 1) {
                    gct = -X_ONE * w[main_id];
                    gcrt = X_ONE * w[main_id];
                    updates[4] = true;
                    updates[5] = true;
                } else if (score_four < 1) {
                    gct = X_THREE * w_t[main_tag_id];
                    gcrt = -X_THREE * w_t[main_tag_id];
                    updates[4] = true;
                    updates[5] = true;
                }
            }

            if (updates[0] && updates[1] && updates[2]) {
                grad_w[main_id] += gw.cwiseProduct(gw);
                grad_c[context_id] += gc.cwiseProduct(gc);
                grad_c[random_id] += gcr.cwiseProduct(gcr);

                w[main_id] -= alpha * gw.cwiseProduct((grad_w[main_id] + one_vect).cwiseInverse().cwiseSqrt());
                c[context_id] -=
                        alpha * gc.cwiseProduct((grad_c[context_id] + one_vect).cwiseInverse().cwiseSqrt());
                c[random_id] -= alpha * gcr.cwiseProduct(
                        (grad_c[random_id] + one_vect).cwiseInverse().cwiseSqrt());

            }
            if (updates[3]) {
                grad_w_t[main_tag_id] += gwt.cwiseProduct(gwt);
                w_t[main_tag_id] -= alpha * gwt.cwiseProduct((grad_w_t[main_tag_id] + one_vect).cwiseInverse().cwiseSqrt());
            }
            if (updates[4] && updates[5]) {
                grad_c_t[context_tag_id] += gct.cwiseProduct(gct);
                c_t[context_tag_id] -=
                        alpha * gct.cwiseProduct((grad_c_t[context_tag_id] + one_vect).cwiseInverse().cwiseSqrt());
                if (tagged_random) {
                    grad_c_t[random_tag_id] += gcrt.cwiseProduct(gcrt);
                    c_t[random_tag_id] -= alpha * gcrt.cwiseProduct(
                            (grad_c_t[random_tag_id] + one_vect).cwiseInverse().cwiseSqrt());
                }
            }
        }
    }
    pthread_exit(NULL);
}


void write_line(ofstream &reps_file, VectorXd vec, string label) {
    reps_file << label + " ";
    for (int i = 0; i < D; ++i)
        reps_file << vec[i] << " ";
    reps_file << endl;
}

void save_model(string fname) {
    ofstream reps_file;
    reps_file.open(fname);

    for (auto x = words.begin(); x != words.end(); ++x) {
        if (contexts.find(*x) != contexts.end())
            write_line(reps_file, 0.5 * (w[*x] + c[*x]), strings[*x].Word);
        else
            write_line(reps_file, w[*x], strings[*x].Word);
    }

    for (auto x = contexts.begin(); x != contexts.end(); ++x) {
        if (words.find(*x) == words.end())
            write_line(reps_file, c[*x], strings[*x].Word);
    }

    for (auto x = tags_words.begin(); x != tags_words.end(); ++x) {
            write_line(reps_file, w_t[*x], "/" + tag_names[*x]);
    }
    for (auto x = tags_contexts.begin(); x != tags_contexts.end(); ++x) {
        if (tags_words.find(*x) == tags_words.end())
            write_line(reps_file, c_t[*x], tag_names[*x]);
    }
    reps_file.close();
}

int calculate_boundaries(size_t num_of_threads, vector<size_t> &word_pairs_index, vector<size_t> &starts,
                         vector<size_t> &ends) {
    random_shuffle(word_pairs_index.begin(), word_pairs_index.end());
    starts.clear();
    ends.clear();
    size_t end_p = 0;
    for (int i = 0; i < num_of_threads; i++) {
        end_p = word_pairs_index.size() / num_of_threads;
        ends.push_back(end_p + ((end_p) * i));
        if (i == 0)
            starts.push_back(0);
        else
            starts.push_back(ends[i - 1] + 1);
    }
    ends[num_of_threads - 1] = ends[num_of_threads - 1] + (word_pairs_index.size() % num_of_threads);
}


void RunEpoch(int no_threads, PairsContent &pairsContent){
    pthread_t *pt = (pthread_t *) malloc(no_threads * sizeof(pthread_t));
    vector<size_t> start_points;
    vector<size_t> end_points;
    calculate_boundaries(no_threads, pairsContent.WordPairsIndex, start_points, end_points);
    vector<TrainArgs> arg_vec(no_threads);
    for (int a = 0; a < no_threads; a++) {
        TrainArgs thread_args;
        thread_args.ThreadId = a;
        thread_args.PairContent = &pairsContent;
        thread_args.StartPoint = start_points[a];
        thread_args.EndPoint = end_points[a];
        arg_vec[a] = thread_args;
    }
    for (int a = 0; a < no_threads; a++) {
        pthread_create(&pt[a], NULL, train, (void *) &arg_vec[a]);
    }
    for (int a = 0; a < no_threads; a++) pthread_join(pt[a], NULL);
    free(pt);
}
int parallel_launch(int no_threads, PairsContent &regularPairs, PairsContent &taggedPairs) {
    cout << "Total epohs to train = " << epohs << endl;
    cout << "Vector Dimension = " << D << endl;
    cout << "Random Sample count = " << sample_count << endl;
    for (auto t = 0; t < epohs; ++t) {
        RunEpoch(no_threads, regularPairs);
        if (!taggedPairs.WordPairs.empty()) {
            RunEpoch(no_threads, taggedPairs);
        }
    }
}

int main(int argc, char *argv[]) {
    no_threads = 100;

    if (argc == 1) {
        fprintf(stderr, "usage: ./sswe_b2 --corpus=cor \
                                --dimension=dim --epohs=iteration_count --sample_count=s_count --output=vectors\n");
        return 0;
    }
    parse_args::init(argc, argv);

    string pair_file = parse_args::get<string>("--pairs");
    string pair_file_tagged = parse_args::get<string>("--pairs-tagged");

    D = parse_args::get<int>("--dimension");
    //cout<<pair_file;
    string unigram_file = parse_args::get<string>("--unigramm");
    string unigram_file_tagged = parse_args::get<string>("--unigramm-tagged");;

    string output_file = parse_args::get<string>("--output");
    epohs = parse_args::get<int>("--epohs");
    cout << "epohs " << epohs << endl;
    alpha = 0.75;
    sample_count = parse_args::get<int>("--sample_count");

    PairsContent regularPairs;
    read_pairs(pair_file, regularPairs.WordPairs, regularPairs.WordPairsIndex);

    PairsContent taggedPairs;
    if (!pair_file_tagged.empty() && !unigram_file_tagged.empty()) {
        read_pairs(pair_file_tagged, taggedPairs.WordPairs, taggedPairs.WordPairsIndex);
    }

    load_unigramT(unigram_file, regularPairs.UnigramTable);
    if (!pair_file_tagged.empty() && !unigram_file_tagged.empty()) {
        load_unigramT(unigram_file_tagged, taggedPairs.UnigramTable);
    }

    cout << endl;
    initialize();
    cout << endl;


    parallel_launch(no_threads, regularPairs, taggedPairs);
    save_model(output_file);
    cout << "Finished..." << endl;

}
