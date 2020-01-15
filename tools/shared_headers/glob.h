#include<vector>
#include<filesystem>
#include<string>

using std::vector;
using std::string;
namespace fs = std::filesystem;

// Returns all files in the directory matching the extension.
vector<fs::path> glob_ext(const fs::path& root, const string& ext){
  vector<fs::path> result;
  for(auto& p_ent: fs::directory_iterator(root)){
    fs::path p = p_ent.path();
    if(p.extension() == ext)
      result.push_back(p);
  }
  return result;
}
