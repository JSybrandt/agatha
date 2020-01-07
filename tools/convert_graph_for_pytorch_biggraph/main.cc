#include<iostream>
#include <highfive/H5Easy.hpp>
#include <CLI/CLI.hpp>
#include <sys/types.h>
#include <dirent.h>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;

static bool endsWith(const std::string& str, const std::string& suffix){
    return str.size() >= suffix.size()
           && 0 == str.compare(str.size()-suffix.size(),
                               suffix.size(),
                               suffix);
}

vector<string> glob_tsv(string root){
  vector<string> result;
  DIR* dir_ptr = opendir(root.c_str());
  struct dirent * dp;
  while ((dp = readdir(dir_ptr)) != NULL) {
    if(endsWith(dp->d_name, ".tsv"))
      result.push_back(dp->d_name);
  }
  closedir(dir_ptr);
  return result;
}


int main(int argc, char **argv){
  //CLI Options
  CLI::App app{
    "This converts a directory of TSV files to pytorch biggraph format"
  };
  string tsv_dir_path;
  app.add_option(
      "-t,--tsv_dir",
      tsv_dir_path,
      "Location of input TSV files."
  );
  string ptbg_dir_path;
  app.add_option(
      "-p,--ptbg_dir",
      ptbg_dir_path,
      "Root directory of output files."
  );
  CLI11_PARSE(app, argc, argv);

  cout << "Getting TSV files from " << tsv_dir_path << endl;
  vector<string> tsv_files = glob_tsv(tsv_dir_path);
  cout << "Found " << tsv_files.size() << endl;

  

  return 0;
}
