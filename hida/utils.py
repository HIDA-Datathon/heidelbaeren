import os

def get_os(name, default=None, required=False):
    os_var = os.environ.get(name)
    if name in ['VAL_FOLDS', 'TRAIN_FOLDS']:
        if os.environ.get(name) is None:
            os_var = []
        else:
            os_var = os.environ.get(name).split(' ')
    if isinstance(os_var, str) and os_var.lower() == 'true':
        os_var = True
    elif isinstance(os_var, str) and os_var.lower() == 'false':
        os_var = False
        
    if os_var is None and default is not None:
        os_var = default
    # if os_var is None and required is True:
    #     raise AssertionError(f'You must define a {name}')
    return os_var