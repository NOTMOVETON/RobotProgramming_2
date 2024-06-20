import yaml

class Params:
    '''
    Class for easy reading and accessing parameters from a YAML file.

    Arguments:
    ----------
    path : str
        Path to the YAML file containing the parameters.

    '''
    def __init__(self, path):
        with open(path) as params_file:
            params = yaml.safe_load(params_file)
            self.__dict__.update(params)