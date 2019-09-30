# PyVista
import pyvista as pv
file_name = directory + filename
mesh = pv.read(file_name)

# Common display argument to make sure all else is constant
dargs = dict(scalars=field_name, cmap='rainbow', show_edges=True)

p = pv.Plotter(shape=(2,2))
p.subplot(0,0)
p.add_mesh(mesh, interpolate_before_map=False,
           stitle='Elevation - not interpolated', **dargs)
p.subplot(1,1)
p.add_mesh(mesh, interpolate_before_map=True,
           stitle='Elevation - interpolated', **dargs)
p.link_views()
p.camera_position = [(0, 0, 2.0),
                     (0.0, 0.0, 0.0),
                     (0.00, 0.37, 0.93)]
p.show()