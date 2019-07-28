#!/usr/bin/env python3
import ray
import time
from pathlib import Path

def get_ray_head_addr():
  host_path = Path("/home/jsybran/.addresses/ray_head")
  assert host_path.is_file()
  with open(host_path) as p:
    return p.read().strip()


ray.init(redis_address=get_ray_head_addr())

@ray.remote
def f():
  time.sleep(0.2)
  return ray.services.get_node_ip_address()

# Get a list of the IP addresses of the nodes that have joined the cluster.
print(set(ray.get([f.remote() for _ in range(1000)])))
