from scipy.interpolate import griddata

def read_vtk(file_name, field_name, filetype="vtu"):
    """Read the data in vtk/vtu/vtp files
    Return a dictionary containing 'mesh' and other physics field data in numpy form
    """

    reader = vtk.vtkXMLUnstructuredGridReader()
    if filetype == "vtp":
        reader = vtk.vtkXMLPolyDataReader()

    reader.SetFileName(file_name)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()

    # Get the field data
    vtk_array_dict = dict()
    numpy_array_dict = dict()

    #field_names = [item for item in field_name]

    #while isinstance(field_names[0], list):
    #    field_names = [item for item in field_names[0]]

    # Get the mesh data
    nodes_vtk_array = reader.GetOutput().GetPoints().GetData()
    numpy_array_dict['mesh'] = vtk_to_numpy(nodes_vtk_array)

    vtk_array_dict[field_name] = output.GetPointData().GetArray(field_name)
    numpy_array_dict[field_name] = vtk_to_numpy(output.GetPointData().GetArray(field_name))

    return numpy_array_dict

def interpolate_field(data, ranges=None, x_points=100, dim=2):
    """Do interpolation in the field using the mesh point data
    to make the a complete data matrix
    2D interpolation is developed now, there are three components:
    vx, vy, v_magnitude
    """

    field_names = [item for item in data.keys()].pop()
    while not isinstance(field_names, list):
        field_names = [field_names]

    if ranges is None:
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
    else:
        xmin, xmax = ranges[0][0], ranges[0][1]
        ymin, ymax = ranges[1][0], ranges[1][1]

    npts_x = x_points
    npts_y = np.floor(npts_x * (ymax - ymin) / (xmax - xmin))
    # define grid
    xi = np.linspace(xmin, xmax, npts_x)
    yi = np.linspace(ymin, ymax, npts_y)

    x, y, z = data['mesh'][:, 0], data['mesh'][:, 1], data['mesh'][:, 2]

    field_value_dict = dict()
    if dim == 2:
        # 2D interpolate
        for name in field_names:
            size = data[name].shape
            data_components = []
            for i in range(size[1] - 1):
                data_components.append(griddata((x, y), data[name][:, i], (xi[None, :], yi[:, None]), method='cubic'))

            magnitude = np.sqrt(np.power(data[name][:, 0], 2) + np.power(data[name][:, 1], 2))
            data_components.append(griddata((x, y), magnitude, (xi[None, :], yi[:, None]), method='cubic'))
            field_value_dict[name] = data_components
    else:
        print("Under construction")
    return field_value_dict

# run PIV on all files in a directory and make gif
def data_from_directory(directory, data_directory, print_directory, field_name):

    vx_field = []
    vy_field = []
    vm_field = []

    # part 1: create list of all image files in the directory

    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.vtu'):
            # part 2: Convert .vtu files to numpy arrays
            temp = read_vtk(directory + filename, field_name, filetype = 'vtu')
            temp2 = interpolate_field(temp, ranges = [[0.5, 1.5], [0.0, 0.6]], x_points = 300)

            # part 3: Save data into numpy arrays
            vx_field.append(temp2[field_name][0])
            vy_field.append(temp2[field_name][1])
            vm_field.append(temp2[field_name][2])
            #plt.figure(figsize = (30,20))
            #plt.imshow(temp[field_name][0], cmap = 'RdBu_r')
            #plt.colorbar()
            #plt.clim(0, 0.04)
            #Prints figure into another directory
            #plt.savefig(print_directory + str(count) + '.png')
            print(count)
            count  = count + 1
            continue
        else:
            continue
    np.save(data_directory + "/vx_field", np.array(vx_field))
    np.save(data_directory + "/vy_field", np.array(vy_field))
    np.save(data_directory + "/vm_field", np.array(vm_field))

# run PIV on all files in a directory and make gif
def synthetic_obs(directory_obs, field_name, mean, variance):

    vx_obs = []
    vy_obs = []
    vm_obs = []

    # part 1: create list of all image files in the directory

    count = 0
    for filename in os.listdir(directory_obs):
        if filename.endswith('.vtu'):
            # part 2: Convert .vtu files to numpy arrays
            temp = read_vtk(directory_obs + filename, field_name, filetype = 'vtu')
            temp = interpolate_field(temp, ranges = [[0.5, 1.5], [0.0, 0.6]], x_points = 300)
            # part 3: Save data into numpy arrays
            vx_obs.append(temp[field_name][0] + np.random.normal(mean, variance, temp[field_name][0].shape))
            vy_obs.append(temp[field_name][1] + np.random.normal(mean, variance, temp[field_name][0].shape))
            vm_obs.append(temp[field_name][2] + np.random.normal(mean, variance, temp[field_name][0].shape))
            print(count)
            count = count + 1
            continue
        else:
            continue
    np.save(directory_obs + "/vx_obs", np.array(vx_obs))
    np.save(directory_obs + "/vy_obs", np.array(vy_obs))
    np.save(directory_obs + "/vm_obs", np.array(vm_obs))

def read_directory_vtu_numpy(directory, print_directory, field_name):
    vx_field = []
    vy_field = []
    vm_field = []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.vtu'):
            ugg = vtktools.vtu(directory + filename)
            test = ugg.GetVectorField(field_name)
            vx_field.append(test[:, 0])
            vy_field.append(test[:, 1])
            vm_field.append(np.sqrt(np.square(test[:, 0]) + np.square(test[:, 1])))
            #print(count)
            count = count + 1
            continue
        else:
            continue

    np.save(print_directory + "/vx_field", np.array(vx_field))
    np.save(print_directory + "/vy_field", np.array(vy_field))
    np.save(print_directory + "/vm_field", np.array(vm_field))
