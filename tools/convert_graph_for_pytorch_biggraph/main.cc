#include <sys/stat.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>


#include <argparse.hpp>
#include <cppitertools/enumerate.hpp>
#include <highfive/H5Attribute.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <nlohmann/json.hpp>
#include <omp.h>

#include "glob.h"
#include "parse_kv_json.h"


using json = nlohmann::json;
// Maps the type char to the std::list of names per part
using Partition = std::unordered_map<char, std::vector<std::list<std::string>>>;
// Source, Target, Relation
using Edge = std::tuple<size_t, size_t, size_t>;
// 2d set of edge std::lists
using Buckets = std::vector<std::vector<std::list<Edge>>>;

namespace fs = std::filesystem;

char get_node_type(const std::string& name){
  //Names should be of form x:data where x is a 1 character type, and data may
  //be anything
  if(name[1] != ':')
    throw std::runtime_error("Invalid node name: " + name);
  return name[0];
}

bool is_name_selected(
    const std::string& name,
    const std::unordered_set<char>& types
){
  return types.find(get_node_type(name)) != types.end();
}

std::unordered_set<std::string> get_all_node_names(
    const std::vector<fs::path>& file_names,
    const std::unordered_set<char>& select_types
){
  // Each thread is going to load a segment of the json files, collecting names

  std::unordered_set<std::string> result;
  int num_finished = 0;
  #pragma omp parallel
  {
    std::unordered_set<std::string> local_result;
    #pragma omp for schedule(dynamic)
    for(size_t i = 0; i < file_names.size(); ++i){
      for (const KVPair& kv : parse_kv_json(file_names[i])){
        if(is_name_selected(kv.key, select_types))
          local_result.insert(kv.key);
        if(is_name_selected(kv.value, select_types))
          local_result.insert(kv.value);
      }
      #pragma omp critical
      {
        ++num_finished;
        std::cout << num_finished << "/" << file_names.size() << std::endl;
      }
    }
    #pragma omp critical
    {
      result.insert(local_result.begin(), local_result.end());
    }
  }
  return result;
}

size_t get_node_partition(const std::string& name, size_t num_partitions){
  return std::hash<std::string>{}(name) % num_partitions;
}


Partition get_empty_partition(
    const std::unordered_set<char>& node_types,
    size_t num_parts
){
  Partition result;
  for(char t : node_types){
    result[t] = std::vector<std::list<std::string>>(num_parts);
  }
  return result;
}

Buckets get_empty_buckets(size_t num_parts){
  return Buckets(num_parts, std::vector<std::list<Edge>>(num_parts));
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


//using Partition = std::unordered_map<char, std::vector<vector<std::string>>>;
Partition partition_nodes(
    const std::vector<std::string>& node_names,
    const std::unordered_set<char>& node_types,
    size_t num_partitions
){
  Partition result = get_empty_partition(node_types, num_partitions);
  #pragma omp parallel
  {
    Partition local_result = get_empty_partition(node_types, num_partitions);
    #pragma omp for
    for(size_t i = 0; i < node_names.size(); ++i){
      try{
        const std::string& node = node_names[i];
        char node_type = get_node_type(node);
        size_t part = get_node_partition(node, num_partitions);
        local_result[node_type][part].push_back(node);
      } catch (const std::runtime_error& err) {
        std::cout << "Encountered an issue:" << err.what() << std::endl;
      }
    }
    #pragma omp critical
    {
      for(char type : node_types){
        for(size_t p = 0; p < num_partitions; ++p){
          std::list<std::string>& res_list = result[type][p];
          std::list<std::string>& loc_list = local_result[type][p];
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
    const std::list<std::string>& nodes
){
  std::stringstream file_name;
  file_name << "entity_count_" << node_type << "_" << partition << ".txt";
  fs::path ptbg_entity_count_path = ptbg_entity_dir / file_name.str();
  std::fstream count_file(ptbg_entity_count_path, std::ios::out);
  count_file << nodes.size();
  count_file.close();
}

void write_json_file(
    const fs::path& ptbg_entity_dir,
    char node_type,
    size_t partition,
    const std::list<std::string>& nodes
){
  std::stringstream file_name;
  file_name << "entity_names_" << node_type << "_" << partition << ".json";
  fs::path ptbg_entity_json_path = ptbg_entity_dir / file_name.str();
  std::fstream json_file(ptbg_entity_json_path, std::ios::out);
  json output = nodes;
  json_file << output;
  json_file.close();
}

Buckets bucket_edges(
    const std::vector<fs::path>& kv_json_paths,
    const std::unordered_map<std::string, size_t>& node2idx,
    const std::unordered_map<std::string, size_t>& relation2idx,
    size_t num_partitions
){
  Buckets result = get_empty_buckets(num_partitions);
  size_t num_finished = 0;
  #pragma omp parallel
  {
    Buckets local_buckets = get_empty_buckets(num_partitions);
    #pragma omp for schedule(dynamic)
    for(size_t i = 0; i < kv_json_paths.size(); ++i){
      for (const KVPair& kv : parse_kv_json(kv_json_paths[i])){
        std::stringstream relation;
        relation << get_node_type(kv.key) << get_node_type(kv.value);
        const auto& relation_index = relation2idx.find(relation.str());
        if (relation_index != relation2idx.end()) {
          local_buckets
            [get_node_partition(kv.key, num_partitions)]
            [get_node_partition(kv.value, num_partitions)]
            .push_back({
              node2idx.at(kv.key),
              node2idx.at(kv.value),
              relation_index->second
            });
        }
      }
      #pragma omp critical
      {
        ++num_finished;
        std::cout << num_finished << "/" << kv_json_paths.size() << std::endl;
      }
    }
    #pragma omp critical
    {
      merge_buckets(result, local_buckets);
    }
  }
  return result;
}

void write_hdf5_edge_list(
    const fs::path& hdf5_path,
    const std::list<Edge>& edges
){
  std::vector<size_t> lhs, rhs, rel;
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
      std::stringstream bucket_file_name;
      bucket_file_name << "edges_" << i << "_" << j << ".h5";
      fs::path edge_bucket_path = ptbg_edge_dir / bucket_file_name.str();
      write_hdf5_edge_list(edge_bucket_path, edge_buckets[i][j]);
      ++num_finished;
      std::cout << num_finished << "/" << num_total << std::endl;
    }
  }
}

std::unordered_set<char> split_node_types(const string& input){
  std::unordered_set<char> res;
  std::stringstream ss(input);
  char c;
  while(ss >> c){
    res.insert(c);
  }
  return res;
}

std::unordered_set<std::string> split_relationships(const string& input){
  std::unordered_set<std::string> res;
  std::stringstream ss(input);
  std::string rel;
  while(ss >> rel){
    res.insert(rel);
  }
  return res;
}

int main(int argc, char **argv){
  argparse::ArgumentParser parser("graph_to_ptbg");
  parser.add_argument("-i", "--kv-json-dir")
        .help("Location containing .jsonfiles. As {'key':..., 'value':...}")
        .action([](const std::string& s){ return fs::path(s); })
        .required();
  parser.add_argument("-o", "--ptbg-dir")
        .help("The location to place pytorch graph data. Will create a dir.")
        .action([](const std::string& s){ return fs::path(s); })
        .required();
  parser.add_argument("-c", "--partition-count")
        .default_value(size_t(100))
        .help("Each entity type will be split into this number of parts.")
        .action([](const std::string& s){ return size_t(std::stoi(s)); });
  parser.add_argument("--types")
        .default_value(std::string("selmnp"))
        .help("List of node types to inclide. Types are one char long. "
              "Default: \"selmnp\""
        );
  parser.add_argument("--relations")
        .default_value(std::string(
            "ss se es sl ls sm ms sn ns sp ps pn np pm mp pl lp pe ep"
        ))
        .help(
          "Relations are 2 chars defining directional edges. Default: "
          "\"ss se es sl ls sm ms sn ns sp ps pn np pm mp pl lp pe ep\""
        );
  // This is a flag
  parser.add_argument("--load-entity-partitions")
        .help("Recover the entity partitions from a previous run.")
        .default_value(false)
        .implicit_value(true);
  try {
    parser.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << parser;
    return 1;
  }
  fs::path json_dir = parser.get<fs::path>("--kv-json-dir");
  fs::path ptbg_root_dir = parser.get<fs::path>("--ptbg-dir");
  size_t num_partitions = parser.get<size_t>("--partition-count");
  std::unordered_set<char> node_types =
    split_node_types(parser.get<std::string>("--types"));
  std::unordered_set<std::string> relation_types =
    split_relationships(parser.get<std::string>("--relations"));
  bool load_entity_partitions = parser.get<bool>("--load-entity-partitions");

  assert(fs::is_directory(json_dir));

  // We don't want to depend on the ordering anymore
  std::vector<std::string> ordered_relations(
      relation_types.begin(),
      relation_types.end()
  );
  std::sort(ordered_relations.begin(), ordered_relations.end());

  std::cout << "Selected Types:";
  for (char t : node_types){
    std::cout << " " << t;
  }
  std::cout << std::endl;

  std::cout << "Selected Relationships:";
  for (const std::string& s : ordered_relations){
    std::cout << " " << s;
  }
  std::cout << std::endl;

  std::cout << "Indexing Relations" << std::endl;
  //Check all the relations are valid
  for(const std::string& rel : ordered_relations){
    assert(rel.size() == 2);
    for(char c : rel){
      // ensures that each character from the relations is actually found in the
      // node types
      assert(node_types.find(c) != node_types.end());
    }
  }
  std::unordered_map<std::string, size_t> relation2idx;
  for(const auto& [idx, rel] : iter::enumerate(ordered_relations)){
    relation2idx[rel] = idx;
  }

  std::cout << "Setting up directories" << std::endl;
  fs::path ptbg_edge_dir = ptbg_root_dir / "edges";
  fs::path ptbg_entity_dir = ptbg_root_dir / "entities";
  fs::create_directories(ptbg_edge_dir);
  fs::create_directories(ptbg_entity_dir);

  std::cout << "Getting txt files from " << json_dir << std::endl;
  std::vector<fs::path> kv_json_paths = glob_ext(json_dir, ".txt");

  // Must find .txt files
  assert(kv_json_paths.size() > 0);

  // This is what we're going to load
  std::unordered_map<std::string, size_t> node2idx;
  if(load_entity_partitions){
    std::cout << "Loading entities to inverted index." << std::endl;
    std::vector<fs::path> json_paths = glob_ext(ptbg_entity_dir, ".json");
    size_t num_finished = 0;
    #pragma omp parallel
    {
      std::unordered_map<std::string, size_t> local_node2idx;
      #pragma omp for schedule(dynamic)
      for(size_t path_idx = 0; path_idx < json_paths.size(); ++path_idx){
        std::fstream json_file(json_paths[path_idx], std::ios::in);
        json input;
        json_file >> input;
        json_file.close();
        for(const auto& [node_idx, node] : iter::enumerate(input)){
          local_node2idx[node] = node_idx;
        }
        #pragma omp critical
        {
          ++num_finished;
          std::cout << num_finished << "/" << json_paths.size() << std::endl;
        }
      }
      #pragma omp critical
      {
        node2idx.merge(local_node2idx);
      }
    }
  } else {

    std::cout << "Getting all node names" << std::endl;
    std::unordered_set<std::string> node_names =
      get_all_node_names(kv_json_paths, node_types);

    std::cout << "Ordering Node Names" << std::endl;
    std::vector<std::string> ordered_node_names;
    ordered_node_names.reserve(node_names.size());
    for(auto& n : node_names)
      ordered_node_names.emplace_back(std::move(n));

    std::cout << "Partitioning Node Names" << std::endl;
    Partition type2part2nodes = partition_nodes(
        ordered_node_names,
        node_types,
        num_partitions
    );

    std::cout << "Writing entity partition files" << std::endl;
    std::vector<char> ordered_node_types(node_types.begin(), node_types.end());
    #pragma omp parallel for collapse(2)
    for(size_t type_idx = 0; type_idx < ordered_node_types.size(); ++type_idx)
      for(size_t part_idx = 0; part_idx < num_partitions; ++part_idx){
        char type = ordered_node_types[type_idx];
        const std::list<std::string>& nodes = type2part2nodes[type][part_idx];
        write_count_file(ptbg_entity_dir, type, part_idx, nodes);
        write_json_file(ptbg_entity_dir, type, part_idx, nodes);
      }

    std::cout << "Creating Inverted Index" << std::endl;
    #pragma omp parallel
    {
      std::unordered_map<std::string, size_t> local_node2idx;
      #pragma omp for collapse(2)
      for(size_t type_idx = 0; type_idx < ordered_node_types.size(); ++type_idx)
        for(size_t part_idx = 0; part_idx < num_partitions; ++part_idx){
          char type = ordered_node_types[type_idx];
          const std::list<std::string>& nodes = type2part2nodes[type][part_idx];
          for(const auto& [idx, name] : iter::enumerate(nodes)){
            local_node2idx[name] = idx;
          }
      }
      #pragma omp critical
      {
        node2idx.merge(local_node2idx);
      }
    }

  } // Constructed entity inverted index

  std::cout << "Indexing Edges" << std::endl;

  Buckets edge_buckets = bucket_edges(
    kv_json_paths,
    node2idx,
    relation2idx,
    num_partitions
  );

  std::cout << "Writing Edge HDF5s" << std::endl;
  write_hdf5_edge_buckets(ptbg_edge_dir, edge_buckets);

  return 0;
}
