#ifndef PARSE_KV_JSON_H
#define PARSE_KV_JSON_H

#include <nlohmann/json.hpp>
#include <list>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

using json = nlohmann::json;
namespace fs = std::filesystem;

struct KVPair {
  KVPair(
      const std::string& k,
      const std::string& v
  ): key(k), value(v) {}
  std::string key;
  std::string value;
};

std::string get_value_string(const json& val){
  if (val.is_object() || val.is_array()){
    return val.dump();
  } else {
    return val.get<std::string>();
  }
}

std::list<KVPair> parse_kv_json(const fs::path& json_path){
  std::list<KVPair> res;
  std::fstream json_file(json_path, std::ios::in);
  std::string line;
  while (getline(json_file, line)){
    json key_value_entry = json::parse(line);
    assert(key_value_entry.find("key") != key_value_entry.end());
    assert(key_value_entry.find("value") != key_value_entry.end());
    assert(key_value_entry["key"].is_string());
    res.emplace_back(
        key_value_entry["key"].get<std::string>(),
        get_value_string(key_value_entry["value"])
    );
  }
  json_file.close();
  return res;
}

void merge_duplicate_keys(std::list<KVPair>& table_entries){
  std::unordered_map<std::string, std::unordered_set<std::string>> key2values;
  for (const auto& entry : table_entries){
    key2values[entry.key].insert(entry.value);
  }
  table_entries.clear();
  for(const auto& [key, value_set] : key2values){
    table_entries.emplace_back(
      key,
      json(value_set).dump()
    );
  }
}

#endif
