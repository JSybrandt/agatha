#include <unordered_map>
#include <argparse.hpp>
#include <sqlite_orm/sqlite_orm.h>
#include <sqlite3.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using NodeIdx = unsigned long long;
using json = nlohmann::json;
namespace fs = std::filesystem;
namespace sql = sqlite_orm;

struct GraphEntry {
  std::string node;
  std::string neighbors;
};

NodeIdx get_idx(
    const std::string& node_name,
    std::unordered_map<std::string, NodeIdx>& node2idx
){
  auto idx_iter = node2idx.find(node_name);
  if(idx_iter == node2idx.end()){
    NodeIdx idx = node2idx.size();
    node2idx[node_name] = idx;
    return idx;
  } else {
    return idx_iter->second;
  }
}

int main(int argc, char** argv){
  argparse::ArgumentParser parser("sqlite_graph_to_edgelist");
  parser.add_argument("-i", "--sqlite_graph")
        .help("Location of graph.sqlite3.")
        .action([](const std::string& s){ return fs::path(s); });
  parser.add_argument("-o", "--edge_list")
        .help("The location to write edge list file")
        .action([](const std::string& s){ return fs::path(s); });
  //parser.add_argument("-n", "--name_list")
        //.help("The location to write names corresponding to each index")
        //.action([](const std::string& s){ return fs::path(s); });

  try {
    parser.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << parser;
    return 1;
  }

  fs::path graph_path = parser.get<fs::path>("--sqlite_graph");
  fs::path edge_path = parser.get<fs::path>("--edge_list");
  //fs::path name_path = parser.get<fs::path>("--name_list");

  assert(fs::exists(graph_path));
  assert(!fs::exists(edge_path));
  //assert(!fs::exists(name_path));

  auto storage = sql::make_storage(
      graph_path,
      sql::make_table(
        "graph",
        sql::make_column("node", &GraphEntry::node),
        sql::make_column("neighbors", &GraphEntry::neighbors)
      )
  );

  auto total_size = storage.count<GraphEntry>();

  unsigned long long count = 0;
  std::unordered_map<std::string, NodeIdx> node2idx;
  std::fstream edge_list(edge_path, std::ios::out);
  for(GraphEntry& entry : storage.iterate<GraphEntry>()){
    json neighbors = json::parse(entry.neighbors);
    NodeIdx node_idx = get_idx(entry.node, node2idx);
    for(const std::string& neigh : neighbors){
      NodeIdx neigh_idx = get_idx(neigh, node2idx);
      edge_list << node_idx << " " << neigh_idx << " " << 1 << std::endl;
    }
    ++count;
    if(count % 10000 == 0)
      std::cout << count << "/" << total_size << std::endl;
  }
  edge_list.close();
  return 0;
}
