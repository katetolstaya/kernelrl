from rtree import index

p = index.Property()
p.dimension = 3
p.dat_extension = 'data'
p.idx_extension = 'index'
idx3d = index.Index('3d_index',properties=p)
idx3d.insert(1, (0, 0, 0, 60, 23.0, 42.0))
print idx3d.intersection( (-1, -1, 0, 62, 22, 43))
