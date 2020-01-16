#include <unordered_map>
#include <list>
#include <unordered_set>
#include <argparse.hpp>
#include <nlohmann/json.hpp>
#include <sqlite_orm/sqlite_orm.h>
#include <sqlite3.h>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <sstream>
#include <cppitertools/zip.hpp>
#include "glob.h"
#include "add_index_to_sqlite.h"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace sql = sqlite_orm;

using Neighbors = std::unordered_set<std::string>;
using Graph = std::unordered_map<std::string, Neighbors>;

struct GraphEntry {
  std::string node;
  std::string neighbors;
};

void merge_graphs(Graph& base_graph, Graph& add_graph){
  for(auto& [node, add_neighbors] : add_graph){
    std::unordered_set<std::string>& base_neighbors = base_graph[node];
    base_neighbors.merge(add_neighbors);
  }
}

Graph parse_tsv(const fs::path& tsv_path){
  Graph res;
  std::fstream tsv_file(tsv_path, std::ios::in);
  std::string line;
  while(getline(tsv_file, line)){
    try{
      std::stringstream tsv_parser(line);
      std::string node1, node2;
      getline(tsv_parser, node1, '\t');
      getline(tsv_parser, node2, '\t');
      res[node1].insert(node2);
    } catch (...) {
      std::cerr << "Encountered an issue with: " << line << std::endl;
    }
  }
  return res;
}

int main(int argc, char** argv){
  argparse::ArgumentParser parser("tsvs_to_sqlite");
  parser.add_argument("-i", "--tsv-dir")
        .help("Location containing tsv files.")
        .action([](const std::string& s){ return fs::path(s); });
  parser.add_argument("-o", "--sqlite")
        .help("The location to write sqlite db")
        .action([](const std::string& s){ return fs::path(s); });

  try {
    parser.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << parser;
    return 1;
  }

  fs::path tsv_dir_path = parser.get<fs::path>("--tsv-dir");
  fs::path sqlite_path = parser.get<fs::path>("--sqlite");

  assert(fs::is_directory(tsv_dir_path));
  assert(!fs::exists(sqlite_path));

  std::vector<fs::path> all_tsv_files = glob_ext(tsv_dir_path, ".tsv");
  assert(all_tsv_files.size() > 0);

  std::cout << "Loading whole graph" << std::endl;
  int num_finished = 0;
  Graph graph;
  #pragma omp parallel for schedule(dynamic)
  for(size_t i = 0; i < all_tsv_files.size(); ++i){
    Graph local_graph = parse_tsv(all_tsv_files[i]);
    #pragma omp critical
    {
      merge_graphs(graph, local_graph);
      ++num_finished;
      std::cout << num_finished << "/" << all_tsv_files.size() << std::endl;
    }
  }

  std::cout << "Ordering nodes" << std::endl;
  std::vector<std::string> nodes;
  nodes.reserve(graph.size());
  for(const auto& [node, neigh] : graph){
    nodes.push_back(node);
  }

  std::cout << "Converting neighborhoods to strings" << std::endl;
  std::vector<std::string> neighborhoods(nodes.size());
  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<nodes.size(); ++i){
    neighborhoods[i] = json(graph[nodes[i]]).dump();
    graph[nodes[i]].clear(); // drop list, keep memory reasonable
  }

  std::cout << "Creating Database" << std::endl;
  auto storage = sql::make_storage(
      sqlite_path,
      sql::make_table(
        "graph",
        sql::make_column("node", &GraphEntry::node),
        sql::make_column("neighbors", &GraphEntry::neighbors)
      )
  );
  storage.sync_schema();

  std::cout << "Writing DB" << std::endl;
  storage.transaction([&]{
      // Both strings
      for(const auto&& [node, neighbors]: iter::zip(nodes, neighborhoods)){
        storage.insert(GraphEntry{node, neighbors});
      }
      return true;
  });

  std::cout << "Adding node index to db" << std::endl;
  add_index_to_sqlite(
      sqlite_path,
      "node_index",
      "graph",
      "node"
  );
  return 0;
}
