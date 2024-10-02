#include <iostream>
#include <vector>
#include <set>
#include <chrono>
#include <cfloat>
#include <string>
#include <tuple>
#include <queue>
#include <fstream>
#include <cstdint>
#include <omp.h>
#include <cnpy.h>
#include <cstdlib> // For std::getenv
#include <nlohmann/json.hpp> // For JSON support (add the nlohmann/json header-only library)

#include <array>
#include <cstddef>
#include <iostream>
#include "DocumentScorer.hpp"

using namespace std;
using json = nlohmann::json; // JSON namespace

// Helper function to get environment variable or return default value
template<typename T>
T get_env_var(const char* env_var, T default_value);

template<>
int get_env_var<int>(const char* env_var, int default_value) {
    const char* value = std::getenv(env_var);
    return value ? std::stoi(value) : default_value;
}

template<>
float get_env_var<float>(const char* env_var, float default_value) {
    const char* value = std::getenv(env_var);
    return value ? std::stof(value) : default_value;
}

template<>
string get_env_var<string>(const char* env_var, string default_value) {
    const char* value = std::getenv(env_var);
    return value ? string(value) : default_value;
}


template<>
size_t get_env_var<size_t>(const char* env_var, size_t default_value) {
    const char* value = std::getenv(env_var);
    return value ? (size_t)std::stoi(value) : default_value;
}

int main(int argc, char **argv)
{
    // Set OpenMP number of threads
    omp_set_num_threads(1);

    // Fetch arguments from environment variables
    int k = get_env_var<int>("K", 10);  // Default value is 10
    float thresh = get_env_var<float>("THRESH", 0.5);  // Default threshold is 0.5
    float thresh_query = get_env_var<float>("THRESH_QUERY", 0.5);  // Default threshold_query is 0.5

    size_t n_doc_to_score = get_env_var<size_t>("N_DOC_TO_SCORE", 1000);  // Default is 1000 docs to score
    size_t nprobe = get_env_var<size_t>("NPROBE", 10);  // Default nprobe is 10
    size_t out_second_stage = get_env_var<size_t>("OUT_SECOND_STAGE", 100);  // Default is 100 candidates in second stage
    string queries_id_file = get_env_var<string>("QUERIES_ID_FILE", "queries.tsv");  // Default queries file
    string index_dir_path = get_env_var<string>("INDEX_DIR_PATH", "index_dir");  // Default index dir
    string alldoclens_path = get_env_var<string>("ALLDOCLENS_PATH", "doclens.npy");  // Default doclens file
    
    string run_outputfile = "run.json";
    string elapsed_times_files = "elapsed_times.json";

    // Load query embeddings
    string queries_path = index_dir_path + "/query_embeddings.npy";
    cnpy::NpyArray queriesArray = cnpy::npy_load(queries_path);

    size_t n_queries = queriesArray.shape[0];
    size_t vec_per_query = queriesArray.shape[1];
    size_t len = queriesArray.shape[2];

    cout << "Dimension: " << len << "\n"
         << "Number of queries: " << n_queries << "\n"
         << "Vector per query: " << vec_per_query << "\n";

    uint16_t values_per_query = vec_per_query * len;
    valType *loaded_query_data = queriesArray.data<valType>();

    // Load qid mapping file
    auto qid_map = load_qids(queries_id_file);
    cout << "queries id loaded\n";

    // Load documents
    DocumentScorer document_scorer(alldoclens_path, index_dir_path, vec_per_query);

    json run_results = json::array();  
    json elapsed_times = json::array();  

    uint64_t total_time = 0;

    cout << "SEARCH STARTED\n";
    for (size_t query_id = 0; query_id < n_queries; query_id++)
    {
        auto start = chrono::high_resolution_clock::now();
        globalIdxType q_start = query_id * values_per_query;

        // PHASE 1: Candidate documents retrieval
        auto candidate_docs = document_scorer.find_candidate_docs(loaded_query_data, q_start, nprobe, thresh);

        // PHASE 2: Candidate document filtering
        auto selected_docs = document_scorer.compute_hit_frequency(candidate_docs, thresh, n_doc_to_score);

        // PHASE 3: Second stage filtering
        auto selected_docs_2nd = document_scorer.second_stage_filtering(loaded_query_data, q_start, selected_docs, out_second_stage);

        // PHASE 4: Document scoring
        auto query_res = document_scorer.compute_topk_documents_selected(loaded_query_data, q_start, selected_docs_2nd, k, thresh_query);

        auto elapsed = chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start).count();
        total_time += elapsed;

        // Write results for each query into JSON
        for (int i = 0; i < k; i++)
        {
            json result;
            result["query_id"] = qid_map[query_id];
            result["doc_id"] = get<0>(query_res[i]);
            result["score"] = get<1>(query_res[i]);
            run_results.push_back(result);
        }

        // Store elapsed time for the query
        elapsed_times.push_back({
            {"query_id", qid_map[query_id]},
            {"elapsed_time_ns", elapsed}
        });
    }

    // Write run.json
    // ofstream run_out_file(run_outputfile);
    // run_out_file << run_results.dump(4);  // Pretty print with indentation of 4 spaces
    // run_out_file.close();

    // // Write elapsed_times.json
    // ofstream elapsed_out_file(elapsed_times_files);
    // elapsed_out_file << elapsed_times.dump(4);
    // elapsed_out_file.close();

    // cout << "Average Elapsed Time per query: " << total_time / n_queries << "\n";

    // Create the main output file linking run.json and elapsed_times.json

    // json run_entry = {
    //     {"id", 0},
    //     {"original_dataset", run_outputfile}
    // };

    // json elapsed_entry = {
    //     {"id", 1},
    //     {"original_dataset", elapsed_times_files}
    // };


    // json main_output = json::array({
    //     {
    //         {"id", 0},
    //         {"original_dataset", run_outputfile}
    //     },
    //     {
    //         {"id", 1},
    //         {"original_dataset", elapsed_times_file}
    //     }
    // });




    ofstream main_out_file("run.json");
    main_out_file << run_results.dump(4);
    main_out_file.close();

    ofstream time_out_file("elapsed_times.json");

    time_out_file << elapsed_times.dump(4);
    time_out_file.close();

    return 0;
}
