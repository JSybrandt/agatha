#!/usr/bin/env python3
from redis import Redis
r = Redis()
print(r.keys()[:10])
