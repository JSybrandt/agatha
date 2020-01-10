#include <argparse.hpp>
#include <highfive/H5Attribute.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <iostream>
#include <list>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <filesystem>
#include "enumerate.h"
#include <tuple>
#include <iterator>


using std::cout;
using std::tuple;
using std::endl;
using std::fstream;
using std::find;
using std::hash;
using std::ios;
using std::list;
using std::stoi;
using std::move;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::runtime_error;
using std::make_move_iterator;

using json = nlohmann::json;
// Maps the type char to the list of names per part
using Partition = unordered_map<char, vector<list<string>>>;
// Source, Target, Relation
using Edge = tuple<size_t, size_t, size_t>;
// 2d set of edge lists
using Buckets = vector<vector<list<Edge>>>;

namespace fs = std::filesystem;

// Parses the json file name (expected to be entity_names_{type}_{part}.json)
tuple<char, size_t> parse_json_filename(const fs::path json_file){
  assert(json_file.extension() == ".json");
  stringstream parser(json_file.stem());
  string entity, names, type;
  size_t part;
  try{
    getline(parser, entity, '_');
    assert(entity == "entity");
    getline(parser, names, '_');
    assert(names == "names");
    getline(parser, type, '_');
    assert(type.size() == 1);
    parser >> part;
  } catch(...){
    throw runtime_error("Invalid file name: " + string(json_file));
  }
  return {type[0], part};
}

// Returns all tsv files in the directory
vector<fs::path> glob_ext(const fs::path& root, const string& ext){
  vector<fs::path> result;
  for(auto& p_ent: fs::directory_iterator(root)){
    fs::path p = p_ent.path();
    if(p.extension() == ext)
      result.push_back(p);
  }
  return result;
}


unordered_set<string> get_all_node_names(const vector<fs::path>& file_names){
  // Each thread is going to load a segment of the tsv files, collecting names

  unordered_set<string> result;
  int num_finished = 0;
  #pragma omp parallel
  {
    unordered_set<string> local_result;
    #pragma omp for schedule(dynamic)
    for(size_t i = 0; i < file_names.size(); ++i){
      fstream tsv_file(file_names[i], ios::in);
      string line;
      while(getline(tsv_file, line)){
        stringstream ss(line);
        for(size_t name_idx = 0; name_idx < 2; ++name_idx){
          string name;
          getline(ss, name, '\t');
          local_result.insert(name);
        }
      }
      tsv_file.close();
      #pragma omp critical
      {
        ++num_finished;
        cout << num_finished << "/" << file_names.size() << endl;
      }
    }
    #pragma omp critical
    {
      result.insert(local_result.begin(), local_result.end());
    }
  }
  return result;
}


char get_node_type(const string& name){
  //Names should be of form x:data where x is a 1 character type, and data may
  //be anything
  if(name[1] != ':')
    throw runtime_error("Invalid node name: " + name);
  return name[0];
}


size_t get_node_partition(const string& name, size_t num_partitions){
  return hash<string>{}(name) % num_partitions;
}


Partition get_empty_partition(const vector<char>& node_types, size_t num_parts){
  Partition result;
  for(char t : node_types){
    result[t] = vector<list<string>>(num_parts);
  }
  return result;
}

Buckets get_empty_buckets(size_t num_parts){
  return Buckets(num_parts, vector<list<Edge>>(num_parts));
}

Buckets& merge_buckets(Buckets& base, Buckets& add){
  size_t num_parts = base.size();
  assert(num_parts == add.size());
  for(size_t i = 0; i < num_parts; ++i){
    assert(num_parts == base[i].size());
    assert(num_parts == add[i].size());
    for(size_t j = 0; j < num_parts; ++j){
      auto& l = base[i][j];
      l.splice(l.end(), add[i][j]);
    }
  }
  return base;
}


//using Partition = unordered_map<char, vector<vector<string>>>;
Partition partition_nodes(
    const vector<string>& node_names,
    const vector<char>& node_types,
    size_t num_partitions
){
  Partition result = get_empty_partition(node_types, num_partitions);
  #pragma omp parallel
  {
    Partition local_result = get_empty_partition(node_types, num_partitions);
    #pragma omp for
    for(size_t i = 0; i < node_names.size(); ++i){
      try{
        const string& node = node_names[i];
        char node_type = get_node_type(node);
        size_t part = get_node_partition(node, num_partitions);
        local_result[node_type][part].push_back(node);
      } catch (const runtime_error& err) {
        cout << "Encountered an issue:" << err.what() << endl;
      }
    }
    #pragma omp critical
    {
      for(char type : node_types){
        for(size_t p = 0; p < num_partitions; ++p){
          list<string>& res_list = result[type][p];
          list<string>& loc_list = local_result[type][p];
          res_list.splice(res_list.end(), loc_list);
        }
      }
    }
  }
  return result;
}


void write_count_file(
    const fs::path& ptbg_entity_dir,
    char node_type,
    size_t partition,
    const list<string>& nodes
){
  stringstream file_name;
  file_name << "entity_count_" << node_type << "_" << partition << ".txt";
  fs::path ptbg_entity_count_path = ptbg_entity_dir / file_name.str();
  fstream count_file(ptbg_entity_count_path, ios::out);
  count_file << nodes.size();
  count_file.close();
}

void write_json_file(
    const fs::path& ptbg_entity_dir,
    char node_type,
    size_t partition,
    const list<string>& nodes
){
  stringstream file_name;
  file_name << "entity_names_" << node_type << "_" << partition << ".json";
  fs::path ptbg_entity_json_path = ptbg_entity_dir / file_name.str();
  fstream json_file(ptbg_entity_json_path, ios::out);
  json output = nodes;
  json_file << output;
  json_file.close();
}

Buckets bucket_edges(
    const vector<fs::path>& tsv_files,
    const unordered_map<string, size_t>& node2idx,
    const unordered_map<string, size_t>& relation2idx,
    size_t num_partitions
){
  Buckets result = get_empty_buckets(num_partitions);
  size_t num_finished = 0;
  #pragma omp parallel
  {
    Buckets local_buckets = get_empty_buckets(num_partitions);
    #pragma omp for schedule(dynamic)
    for(size_t tsv_idx = 0; tsv_idx < tsv_files.size(); ++tsv_idx){
      fstream tsv_file(tsv_files[tsv_idx], ios::in);
      string line;
      while(getline(tsv_file, line)){
        try{
          stringstream tsv_parser(line);
          string rel = "__";
          vector<size_t> parts = {0, 0}, local_indices = {0, 0};
          for(size_t node_idx = 0; node_idx < 2; ++node_idx){
            string node_name;
            getline(tsv_parser, node_name, '\t');
            rel[node_idx] = get_node_type(node_name);
            parts[node_idx]  = get_node_partition(node_name, num_partitions);
            local_indices[node_idx] = node2idx.at(node_name);
          }
          const auto& rel_itr = relation2idx.find(rel);
          if(rel_itr != relation2idx.end()){
            local_buckets[parts[0]][parts[1]]
              .push_back({local_indices[0], local_indices[1], rel_itr->second});
          }
        } catch (const runtime_error& err){
          cout << "Encountered an issue with: " << err.what() << endl;
        } catch (const std::out_of_range& err){
          cout << err.what() << endl;
          cout << "Failed to identify a node:" << endl;
          cout << line << endl;
        }
      }
      #pragma omp critical
      {
        ++num_finished;
        cout << num_finished << "/" << tsv_files.size() << endl;
      }
    }
    #pragma omp critical
    {
      merge_buckets(result, local_buckets);
    }
  }
  return result;
}

void write_hdf5_edge_list(const fs::path& hdf5_path, const list<Edge>& edges){
  vector<size_t> lhs, rhs, rel;
  for(auto [s, t, r] : edges){
    lhs.push_back(s);
    rhs.push_back(t);
    rel.push_back(r);
  }
  #pragma omp critical
  {
    HighFive::File h5_file(
        hdf5_path, HighFive::File::OpenOrCreate | HighFive::File::Overwrite
    );
    h5_file.createAttribute<size_t>("format_version", 1);
    HighFive::DataSet lhs_ds = h5_file.createDataSet<size_t>(
        "/lhs", HighFive::DataSpace::From(lhs)
    );
    HighFive::DataSet rhs_ds = h5_file.createDataSet<size_t>(
        "/rhs", HighFive::DataSpace::From(rhs)
    );
    HighFive::DataSet rel_ds = h5_file.createDataSet<size_t>(
        "/rel", HighFive::DataSpace::From(rel)
    );
    lhs_ds.write(lhs);
    rhs_ds.write(rhs);
    rel_ds.write(rel);
  }
}
void write_hdf5_edge_buckets(
    const fs::path& ptbg_edge_dir,
    const Buckets& edge_buckets
){
  size_t num_partitions = edge_buckets.size();
  size_t num_finished = 0, num_total = num_partitions * num_partitions;
  #pragma omp parallel for collapse(2) schedule(dynamic)
  for(size_t i = 0; i < num_partitions; ++i){
    for(size_t j = 0; j < num_partitions; ++j){
      stringstream bucket_file_name;
      bucket_file_name << "edges_" << i << "_" << j << ".h5";
      fs::path edge_bucket_path = ptbg_edge_dir / bucket_file_name.str();
      write_hdf5_edge_list(edge_bucket_path, edge_buckets[i][j]);
      ++num_finished;
      cout << num_finished << "/" << num_total << endl;
    }
  }
}


int main(int argc, char **argv){
  argparse::ArgumentParser parser("tsvs_to_ptbg");
  parser.add_argument("-i", "--tsv-dir")
        .help("Location containing .tsv files. As source[\\t]target[\\t]weight")
        .action([](const std::string& s){ return fs::path(s); });
  parser.add_argument("-o", "--ptbg-dir")
        .help("The location to place pytorch graph data. Will create a dir.")
        .action([](const std::string& s){ return fs::path(s); });
  parser.add_argument("-c", "--partition-count")
        .default_value(size_t(100))
        .help("Each entity type will be split into this number of parts.")
        .action([](const std::string& s){ return size_t(stoi(s)); });
  parser.add_argument("--types")
        .default_value(vector<char>{'s', 'e', 'l', 'm', 'n'})
        .help("List of character names used as node types.");
  parser.add_argument("--relations")
        .default_value(vector<string>{
          "le", "lm", "ln", "ls", "em", "en", "es", "mn", "ms", "ns", "ss"
        })
        .help("List of character names used as node types.");
  parser.add_argument("--load-entity-partitions")
        .help("Recover the entity partitions from a previous run.")
        .default_value(false)
        .implicit_value(true);
  try {
    parser.parse_args(argc, argv);
  }
  catch (const runtime_error& err) {
    cout << err.what() << endl;
    cout << parser;
    return 1;
  }
  fs::path tsv_dir_path = parser.get<fs::path>("--tsv-dir");
  fs::path ptbg_root_dir = parser.get<fs::path>("--ptbg-dir");
  size_t num_partitions = parser.get<size_t>("--partition-count");
  vector<char> node_types = parser.get<vector<char>>("--types");
  vector<string> relation_types = parser.get<vector<string>>("--relations");
  bool load_entity_partitions = parser.get<bool>("--load-entity-partitions");

  assert(fs::is_directory(tsv_dir_path));

  cout << "Indexing Relations" << endl;
  //Check all the relations are valid
  for(const string& rel : relation_types){
    assert(rel.size() == 2);
    for(char c : rel){
      assert(find(node_types.begin(), node_types.end(), c) != node_types.end());
    }
  }
  unordered_map<string, size_t> relation2idx;
  for(const auto& [idx, rel] : enumerate(relation_types)){
    relation2idx[rel] = idx;
  }

  cout << "Setting up directories" << endl;
  fs::path ptbg_edge_dir = ptbg_root_dir / "edges";
  fs::path ptbg_entity_dir = ptbg_root_dir / "entities";
  fs::create_directories(ptbg_edge_dir);
  fs::create_directories(ptbg_entity_dir);

  cout << "Getting TSV files from " << tsv_dir_path << endl;
  vector<fs::path> tsv_files = glob_ext(tsv_dir_path, ".tsv");

  // This is what we're going to load
  unordered_map<string, size_t> node2idx;
  if(load_entity_partitions){
    cout << "Loading entities to inverted index." << endl;
    vector<fs::path> json_paths = glob_ext(ptbg_entity_dir, ".json");
    size_t num_finished = 0;
    #pragma omp parallel
    {
      unordered_map<string, size_t> local_node2idx;
      #pragma omp for schedule(dynamic)
      for(size_t path_idx = 0; path_idx < json_paths.size(); ++path_idx){
        fstream json_file(json_paths[path_idx], ios::in);
        json input;
        json_file >> input;
        json_file.close();
        for(const auto& [node_idx, node] : enumerate(input)){
          local_node2idx[node] = node_idx;
        }
        #pragma omp critical
        {
          ++num_finished;
          cout << num_finished << "/" << json_paths.size() << endl;
        }
      }
      #pragma omp critical
      {
        node2idx.merge(local_node2idx);
      }
    }
  } else {

    cout << "Getting all node names" << endl;
    unordered_set<string> node_names = get_all_node_names(tsv_files);

    cout << "Ordering Node Names" << endl;
    vector<string> ordered_node_names;
    ordered_node_names.reserve(node_names.size());
    for(auto& n : node_names)
      ordered_node_names.emplace_back(move(n));

    cout << "Partitioning Node Names" << endl;
    Partition type2part2nodes = partition_nodes(
        ordered_node_names,
        node_types,
        num_partitions
    );

    cout << "Writing entity partition files" << endl;
    #pragma omp parallel for collapse(2)
    for(size_t type_idx = 0; type_idx < node_types.size(); ++type_idx)
      for(size_t part_idx = 0; part_idx < num_partitions; ++part_idx){
        char type = node_types[type_idx];
        const list<string>& nodes = type2part2nodes[type][part_idx];
        write_count_file(ptbg_entity_dir, type, part_idx, nodes);
        write_json_file(ptbg_entity_dir, type, part_idx, nodes);
      }

    cout << "Creating Inverted Index" << endl;
    #pragma omp parallel
    {
      unordered_map<string, size_t> local_node2idx;
      #pragma omp for collapse(2)
      for(size_t type_idx = 0; type_idx < node_types.size(); ++type_idx)
        for(size_t part_idx = 0; part_idx < num_partitions; ++part_idx){
          char type = node_types[type_idx];
          const list<string>& nodes = type2part2nodes[type][part_idx];
          for(const auto& [idx, name] : enumerate(nodes)){
            local_node2idx[name] = idx;
          }
      }
      #pragma omp critical
      {
        node2idx.merge(local_node2idx);
      }
    }

  } // Constructed entity inverted index

  cout << "Indexing Edges" << endl;

  Buckets edge_buckets = bucket_edges(
    tsv_files,
    node2idx,
    relation2idx,
    num_partitions
  );

  cout << "Writing Edge HDF5s" << endl;
  write_hdf5_edge_buckets(ptbg_edge_dir, edge_buckets);

  return 0;
}
