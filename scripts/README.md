These scripts are used to launch requisite compute resources on the Palmetto
cluster.

 - `start_redis_server`: This script launches redis using the moliere.db.conf.
   This redis DB is intended to be the primary moliere object store. This DB
   must be active for constructing or querying the network.
 - `start_ray_cluster`: This script begins a ray head on the current node, and
   configures all other machines listed in `$PBS_NODEFILE` as raw workers.
 - `stop_ray_cluster`: Closes the processes started by `start_ray_cluster`.

