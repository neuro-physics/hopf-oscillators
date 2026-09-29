import os
import re
import copy
import glob
import argparse
import collections.abc
import scipy.io
import numpy
import numpy.core.records
from enum import IntEnum

#from modules.mouse_track_helper_func_class import structtype

"""

HOW TO ADD A NEW SIMULATION PARAMETER

1. add to one of the functions below ('add_..._params')

2. if you need to pass it to a simulation,
    a) add it to sorw.get_simulation_params or sorw.get_randomwalk_params
    b) add it to the return of sorw.get_simulation_params or sorw.get_randomwalk_params
    c) fix the sorw.get_simulation_params or sorw.get_randomwalk_params
       return in the sorw.Teach_SORW function

       

"""



def add_simulation_params(parser,**defaultValues):
    parser.add_argument('-ntrials'           , nargs=1, required=False, metavar='INT'  , type=int    , default=get_param_value('ntrials'           , defaultValues, [10])   , help='Number of trials to repeat the simulation')
    parser.add_argument('-tTrans'            , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('tTrans'            , defaultValues, [50.0]) , help='[[ NOT IMPLEMENTED ]] units: dt. Transient time to discard before measurements')
    parser.add_argument('-tTotal'            , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('tTotal'            , defaultValues, [200.0]), help='units: dt. Total simulation time')
    parser.add_argument('-dt'                , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('dt'                , defaultValues, [0.01]) , help='Integration time step')
    parser.add_argument('-rngseed'           , nargs=1, required=False, metavar='INT'  , type=int    , default=get_param_value('rngseed'           , defaultValues, [-1])   , help='set to a positive number for it to be the seed of the random number generator, and results be reproducible')
    parser.add_argument('-n_parallel_threads', nargs=1, required=False, metavar='INT'  , type=int    , default=get_param_value('n_parallel_threads', defaultValues, [10])   , help='Number of parallel threads to run (each trial is run in an independent thread)')

    # model parameters
    parser.add_argument('-Z0'            , nargs=1, required=False, metavar='FLOAT', type=complex, default=get_param_value('Z0'            , defaultValues, [0.0]), help='Mean of the initial condition distribution')
    parser.add_argument('-Z0_std'        , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('Z0_std'        , defaultValues, [0.1]), help='Half width of uniform distribution for initial conditions')
    parser.add_argument('-Z_amp_escape'  , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('Z_amp_escape'  , defaultValues, [1.0]), help='Amplitude of Z(t) for measuring escape times')

    parser.add_argument('-alpha'              , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('alpha'         , defaultValues, [0.1])   , help='White noise intensity (standard deviation)')
    parser.add_argument('-beta'               , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('beta'          , defaultValues, [0.001]) , help='Global coupling strength (it can be normalized by total input coupling weight of a node if -normalizecoupling is set)')
    parser.add_argument('-omega'              , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('omega'         , defaultValues, [5.0])   , help='Mean natural frequency of oscillators')
    parser.add_argument('-omega_range'        , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('omega_range'   , defaultValues, [0.05])  , help='(set to 0 for homogeneous parameter) Half width of uniform distribution for natural frequencies')
    parser.add_argument('-lmbda'              , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('lmbda'         , defaultValues, [0.6])   , help='Mean distance to Hopf bifurcation. If input_var_lmbda is set, this is the mean value of lmbda, corresponding to input_var_lmbda = 0.')
    parser.add_argument('-lmbda_range'        , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('lmbda_range'   , defaultValues, [0.10])  , help='(set t0 0 for homogeneous parameter) Half width of uniform distribution for lmbda. If input_var_lmbda is set, this is the range within which input_var_lmbda will be distributed: lmbda +- lmbda_range')
    parser.add_argument('-lmbda_sigmoid_map_k', nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('lmbda'         , defaultValues, [0.6])   , help='If input_var_lmbda is set, this is the slope of the sigmoid map from input_var_lmbda to lmbda parameter. It determines how steeply separated are negative from positive input_var_lmbda values')
    parser.add_argument('-v_tract'            , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('v_tract'       , defaultValues, [5.0])   , help='conduction speed in tract (this parameter is used to calculate input delays as T_ij = Fiber Length_ij / v_tract)')

    # general parameters
    parser.add_argument('-integrator'            , nargs=1, required=False, metavar='STR', type=str,   default=get_param_value('integrator'         , defaultValues,['Euler'])      , choices=['Euler','RK2'], help='integrator used for oscillators')
    parser.add_argument('-outputFilePrefix'      , nargs=1, required=False, metavar='STR', type=str,   default=get_param_value('outputFilePrefix'   , defaultValues,['osc'])        , help='prefix of the output file name')
    parser.add_argument('-input_code'            , nargs=1, required=False, metavar='STR', type=str,   default=get_param_value('input_code'         , defaultValues,[''])           , help='code of the individual to simulate')
    parser.add_argument('-input_dir'             , nargs=1, required=False, metavar='STR', type=str,   default=get_param_value('input_dir'          , defaultValues,['']   )        , help='directory where input matrices are stored')
    parser.add_argument('-input_var_FL'          , nargs=1, required=False, metavar='STR', type=str,   default=get_param_value('input_var_FL'       , defaultValues,['M_FL'])       , help='name of the variable containing the matrix for fiber length in the mat file')
    parser.add_argument('-input_var_FN'          , nargs=1, required=False, metavar='STR', type=str,   default=get_param_value('input_var_FN'       , defaultValues,['M_FN'])       , help='name of the variable containing the matrix for fiber number in the mat file')
    parser.add_argument('-input_var_FMRI'        , nargs=1, required=False, metavar='STR', type=str,   default=get_param_value('input_var_FMRI'     , defaultValues,['M_FMRI'])     , help='name of the variable containing the matrix for rs-fMRI correlation matrix in the mat file')
    parser.add_argument('-input_var_lmbda'       , nargs=1, required=False, metavar='STR', type=str,   default=get_param_value('input_var_lmbda'    , defaultValues, [''])          ,
        help='name of the variable containing the vector used to normalize each lambda_i parameter for oscillator i; e.g., the z-score of node volume (Hutchings 2015 used node surface area). If set, lambda_i is sigmoidally mapped according to this variable: the more negative the value for oscillator i, the closer lambda_i is to the Hopf bifurcation (lambda_i = 1), while larger values map to smaller lambda_i, farther from the bifurcation. The mapping is controlled by lmbda (mean value corresponding to input_var_lmbda=0), lmbda_range (defining max and min lambda by lmbda +- lmbda_range), and lmbda_sigmoid_map_k (the slope of the sigmoid map).'
    )

    # simulation flags
    parser.add_argument('-stopAtEscape'          , required=False, action='store_true', default=False, help='If true, stops simulation when all oscillators escaped the resting state. It can improve simulation time.')
    parser.add_argument('-normalizecoupling'     , required=False, action='store_true', default=False, help='If true, divides coupling beta by the sum of input weights')
    parser.add_argument('-savealltau'            , required=False, action='store_true', default=False, help='If true, saves all escape times of all trials for all nodes (OUTPUT MEMORY = 2*8*n_subjects*ntrials*N_nodes ~ 300MB; SIMULATION MEMORY = 2*8*n_subjects*(tTotal+max_delay)/dt) ~ 60 MB; for 60 subjects, 1000 trials and 306 nodes;')
    parser.add_argument('-writeOnRun'            , required=False, action='store_true', default=False, help='[[ NOT IMPLEMENTED ]] If true, writes output during run')
    parser.add_argument('-testrun'               , required=False, action='store_true', default=False, help='If true, runs only 1 subject and saves the Z variable for debug purpose (e.g., for checking if network activity diverged)')

    return parser


def get_simParam_struct_validate_args(args):
    s                     = namespace_to_structtype(args) # fix scalar input parameters automatically in this conversion
    s.ntrials             = input_positive_int(s.ntrials)
    s.n_parallel_threads  = input_positive_int(s.n_parallel_threads)
    s.rngseed             = input_int_in_range( 'rngseed'              , s.rngseed            ,-1)
    s.tTrans              = input_float_in_range('tTrans'              , s.tTrans             , 0.0)
    s.tTotal              = input_float_in_range('tTotal'              , s.tTotal             , s.tTrans)
    s.dt                  = input_float_in_range('dt'                  , s.dt                 , 0.0)
    s.Z0_std              = input_float_in_range('Z0_std'              , s.Z0_std             , 0.0)
    s.Z_amp_escape        = input_float_in_range('Z_amp_escape'        , s.Z_amp_escape       , 0.0)
    s.v_tract             = input_float_in_range('v_tract'             , s.v_tract            , 0.0)
    s.lmbda_sigmoid_map_k = input_float_in_range('lmbda_sigmoid_map_k' , s.lmbda_sigmoid_map_k, 0.0)
    s.omega_range         = input_float_in_range('omega_range'         , s.omega_range        , 0.0)
    s.lmbda_range         = input_float_in_range('lmbda_range'         , s.lmbda_range        , 0.0)
    s.integrator          = int(IntegratorType[s.integrator])
    assert os.path.isdir(s.input_dir), f'*** invalid input dir: {s.input_dir}'
    assert len(s.input_var_FN)   > 0, '*** input_var_FN must have the name of the variable contaning the fiber number matrix in each input file'
    assert len(s.input_var_FL)   > 0, '*** input_var_FL must have the name of the variable contaning the fiber length matrix in each input file'
    assert len(s.input_var_FMRI) > 0, '*** input_var_FMRI must have the name of the variable contaning the fMRI matrix in each input file'
    assert len(s.input_var_lmbda)>=0, '*** input_var_lmbda must either be empty (then lambda is a random parameter) or have the name of the variable contaning the lambda values that will be mapped to the oscillator using a sigmoid function'
    return s

"""
####################################
#################################### 
####################################
#################################### Input parameter assignments
####################################
#################################### 
####################################
"""

def input_positive_int(value):
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")

def input_nonnegative_int(value):
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")

def input_odd_positint(value):
    n = input_positive_int(value)
    if (n%2) == 0:
        raise argparse.ArgumentTypeError(f"{value} is not an odd positive integer")
    return n

def input_int_in_range(argname,value,a,b=None):
    b = numpy.inf if type(b) is type(None) else b
    try:
        ivalue = int(value)
        if not((ivalue >= a) and (ivalue <= b)):
            raise argparse.ArgumentTypeError(f"{argname} must be in the range [{a};{b}] (inclusive)")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{argname} is not a valid integer")

def input_float_in_range(argname,value,a,b=None):
    b = numpy.inf if type(b) is type(None) else b
    try:
        ivalue = float(value)
        if not((ivalue >= a) and (ivalue <= b)):
            raise argparse.ArgumentTypeError(f"{argname} must be in the range [{a};{b}] (inclusive)")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{argname} is not a valid float")

def get_new_file_name(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.isfile(path):
        path = filename + "_" + str(counter) + extension
        counter += 1
    return path

def _exists(X):
    return not(type(X) is type(None))

def get_output_filename(simParam,N=None,suffix=''):
    fprefix, fext  = os.path.splitext(simParam.outputFilePrefix)
    fext           = '.mat'                  if ((len(fext)==0) or (fext != '.mat')) else fext
    N_txt          = f'_N{N}' if _exists(N) else ''
    fn             = fprefix + N_txt + f'_ntrials{simParam.ntrials}' + f'_tTotal{simParam.tTotal}' +\
                     f'_dt{simParam.dt}' + f'_alpha{simParam.alpha}' + f'_beta{simParam.beta}' +\
                     f'_w{simParam.omega}' + f'_lmbda{simParam.lmbda}' + f'_Zth{simParam.Z_amp_escape}' + suffix + fext
    return fn

def convert_structarray_to_struct_of_arrays(traj):
    N = len(traj)
    if N > 1:
        traj_s = structtype(**{k:[] for k in traj[0].keys()})
        for i in range(N):
            for k,v in traj[i].items():
                traj_s[k].append(v)
        for k,v in traj_s.items():
            if _not_iterable_or_str(v[0]):
                traj_s[k] = numpy.asarray(v)
        return traj_s
    else:
        return traj

def _not_iterable_or_str(v):
    is_iter = isinstance(v,collections.abc.Iterable)
    is_str  = type(v) is str
    return (not is_iter) or (is_iter and is_str)


def convert_struct_of_arrays_to_structarray(sites):
    N = len(sites[sites.GetFields(',').split(',')[0]])
    if N > 1:
        sites_structarray = [ structtype(**{k:None for k in sites.keys()}) for _ in range(N) ]
        for i in range(N):
            for k in sites.keys():
                sites_structarray[i][k] = sites[k][i]
            #sites_structarray[i].x     = sites.x[i]
            #sites_structarray[i].neigh = sites.neigh[i]
            #sites_structarray[i].p     = sites.p[i]
            #sites_structarray[i].n     = sites.n[i]
        return sites_structarray
    else:
        return sites


def convert_items_structtype_to_scipy_struct_array(s,is_scalar_struct=True):
    for k in s.keys():
        if isinstance(s[k],structtype):
            s[k] = convert_structtype_to_scipy_struct_array(s[k],is_scalar_struct=is_scalar_struct)
    return s

def convert_structtype_to_scipy_struct_array(s,is_scalar_struct=True):
    if is_scalar_struct:
        return struct_scalar_for_scipy(list(s.keys()),*s.values())
    else:
        return struct_array_for_scipy(list(s.keys()),*s.values())

def struct_scalar_for_scipy(field_names, *fields_data):
    """
    Return a scalar NumPy record representing a MATLAB struct whose fields
    contain arbitrary NumPy arrays or other objects.
    """
    fn_list = field_names.split(',') if isinstance(field_names, str) else field_names
    assert len(fn_list) == len(fields_data), 'you must give one field name for each field data'
    dtype = [(name, object) for name in fn_list]
    s = numpy.empty((), dtype=dtype)
    for name, data in zip(fn_list, fields_data):
        s[name] = data
    return s

def struct_array_for_scipy(field_names,*fields_data):
    """
    returns a data structure which savemat in scipy.io interprets as a MATLAB struct array
    the order of field_names must match the order in which the remaining arguments are passed to this function
    such that
    s(j).(field_names(i)) == fields_data[i][j], identified by field_names[i]

    field_names ->  comma-separated string listing the field names OR list of str fieldnames;
                        'field1,field2,...' -> field_names(1) == 'field1', etc...
    fields_data ->  each extra argument entry is a list with the data for each field of the struct
                        fields_data[i][j] :: data for field i in the element j of the struct array: s(j).(field_names(i))
    
    returns
        numpy record array S where
        S[field_names[i]][j] == fields_data[i][j]
    """
    fn_list = field_names.split(',') if isinstance(field_names,str) else field_names
    assert len(fn_list) == len(fields_data),'you must give one field name for each field data'
    return numpy.core.records.fromarrays([f for f in fields_data],names=fn_list,formats=[object]*len(fn_list))

def list_of_arr_to_arr_of_obj(X):
    n = len(X)
    Y = numpy.empty((n,),dtype=object)
    for i,x in enumerate(X):
        Y[i] = x
    return Y

def fix_output_fileName(outputFileName,remove_existing_output=True):
    """fix output file extension and remove output files if they already exist, creates output directory if they don't exist"""
    if outputFileName.lower().endswith('.txt'):
        outputFileName = outputFileName.replace('.txt','.mat')
    if not outputFileName.lower().endswith('.mat'):
        outputFileName += '.mat'

    if remove_existing_output:
        if os.path.isfile(outputFileName):
            print("* Replacing ... %s" % outputFileName)
            os.remove(outputFileName)
    
    if has_dir_in_path(outputFileName):
        d = os.path.split(outputFileName)[0]
        if d:
            os.makedirs(d,exist_ok=True)

    return outputFileName

def has_dir_in_path(path):
    return ('/' in path) or ('\\' in path)

def fix_args_lists_as_scalars(args,return_type_for_values=None):
    if type(args) is dict:
        a = args
    else:
        a = args.__dict__
    for k,v in a.items():
        if (not numpy.isscalar(v)) and (len(v) == 1): #(type(v) is list) and (len(v) == 1):
            a[k] = v[0]
        if not( type(return_type_for_values) is type(None)):
            a[k] = return_type_for_values(a[k])
    if type(args) is dict:
        args = a
    else:
        args.__dict__ = a
    return args

def get_param_range(args):
    if args.parScale[0] == 'log':
        v1 = args.parVal1[0]
        v2 = args.parVal2[0]
        if numpy.sign(v1) != numpy.sign(v2):
            raise ValueError('the signs of parVal1 and parVal2 must be the same')
        s = float(numpy.sign(v1))
        return s*numpy.logspace(numpy.log10(v1),numpy.log10(v2),args.nPar[0])
    elif args.parScale[0] == 'linear':
        return numpy.linspace(args.parVal1[0],args.parVal2[0],args.nPar[0])
    else:
        raise ValueError('unknown parScale')

def get_param_value(paramName,args,default):
    if paramName in args.keys():
        return args[paramName]
    return default

def namespace_to_structtype(a,return_type_for_values=None):
    return structtype(**fix_args_lists_as_scalars(copy.deepcopy(a.__dict__),return_type_for_values=return_type_for_values))

def _recordarray_to_structtype(r):
    unpack_array = lambda a: a if numpy.isscalar(a) else (a.item(0) if a.size==1 else a)
    return structtype(**{ field : unpack_array(r[field]) for field in r.dtype.names })

def get_filename_no_ext(file_list):
    if isinstance(file_list,str):
        return os.path.splitext(os.path.split(file_list)[-1])[0]
    else:
        return [ get_filename_no_ext(f) for f in file_list ]

def get_subject_code(s):
    """
    Extracts '0ddd_d' or 'ddd_d' from patterns '_0ddd_d' or '_ddd_d'.
    Returns the extracted string, or None if no match is found.
    """
    if isinstance(s, str):
        match = re.search(r'_?(0?\d{3}_\d)(?:_|\.|\b)', s)
        return match.group(1) if match else None
    else:
        return [get_subject_code(ss) for ss in s]

#def get_input_txt_file_lists(FL_dir,FN_dir,FMRI_dir):
#    file_list_FL   = glob.glob(os.path.join(FL_dir   ,'*.txt'))
#    file_list_FN   = glob.glob(os.path.join(FN_dir   ,'*.txt'))
#    file_list_FMRI = glob.glob(os.path.join(FMRI_dir ,'*.txt'))
#
#    codes_FL       = set(get_subject_code(get_filename_no_ext(file_list_FL)))
#    codes_FN       = set(get_subject_code(get_filename_no_ext(file_list_FN)))
#    codes_FMRI     = set(get_subject_code(get_filename_no_ext(file_list_FMRI)))
#    codes_valid    = sorted(list(codes_FL & codes_FN & codes_FMRI))
#    codes_sort_map = { c:k for k,c in enumerate(codes_valid) }
#
#    file_list_FL_valid   = sorted([ f for f in file_list_FL   if get_subject_code(get_filename_no_ext(f)) in codes_valid ],key=lambda f: codes_sort_map[get_subject_code(get_filename_no_ext(f))])
#    file_list_FN_valid   = sorted([ f for f in file_list_FN   if get_subject_code(get_filename_no_ext(f)) in codes_valid ],key=lambda f: codes_sort_map[get_subject_code(get_filename_no_ext(f))])
#    file_list_FMRI_valid = sorted([ f for f in file_list_FMRI if get_subject_code(get_filename_no_ext(f)) in codes_valid ],key=lambda f: codes_sort_map[get_subject_code(get_filename_no_ext(f))])
#    return codes_valid,file_list_FL_valid,file_list_FN_valid,file_list_FMRI_valid
#
#def get_input_matrices_txt(input_txt_file_FL,input_txt_file_FN,input_txt_file_FMRI):
#    return numpy.loadtxt(input_txt_file_FL),numpy.loadtxt(input_txt_file_FN),numpy.loadtxt(input_txt_file_FMRI)

def get_input_mat_file_list(MAT_dir):
    fl             = glob.glob(os.path.join(MAT_dir,'*.mat'))
    codes          = sorted(get_subject_code(get_filename_no_ext(fl)))
    codes_sort_map = { c:k for k,c in enumerate(codes) }
    return codes, sorted(fl,key=lambda f:codes_sort_map[get_subject_code(get_filename_no_ext(f))])

def get_input_matrices_mat(input_mat_file,varname_FL,varname_FN,varname_FMRI,varname_lmbda_feature=''):
    d          = scipy.io.loadmat(input_mat_file,squeeze_me=True)
    try:
        lmbda_feat = d[varname_lmbda_feature] if varname_lmbda_feature in d else numpy.empty(0,dtype=float)
        return d[varname_FL],d[varname_FN],d[varname_FMRI],lmbda_feat
    except KeyError as e:
        raise ValueError(f'*** variable not found: {e} is not found in {input_mat_file}') from None

def select_code(code,file_list,return_as_list=True):
    if isinstance(code, str):
        for f in file_list:
            if code == get_subject_code(get_filename_no_ext(f)):
                return [f] if return_as_list else f
        return [] if return_as_list else None
    else:
        assert isinstance(code,list), 'code is either an str or a list of str'
        r = []
        for c in code:
            f = select_code(c,file_list,return_as_list=False)
            if f:
                r.append(f)
        return r

def select_code_or_die(code,file_list):
    if (not isinstance(code,type(None))) and len(code):
        file_list = select_code(code,file_list,return_as_list=True)
        assert len(file_list), f'*** selected code not found: input_code == {code}'
    return file_list

#def load_escape_times_file(fnames):
#    d = { k:(v if k=='codes' else _recordarray_to_structtype(v)) for k,v in scipy.io.loadmat(fnames,squeeze_me=True).items() if (k[:2]!='__') and (k[-2:]!='__') }
#    return d

class IntegratorType(IntEnum):
    Euler = 0
    RK2   = 1

def load_escape_times_file(*fnames):
    d = scipy.io.loadmat(*fnames, squeeze_me=True, struct_as_record=False)
    return {
        k: matstruct_to_structtype(v)
        for k, v in d.items()
        if not (k[:2] == '__' or k[-2:] == '__')
    }

def matstruct_to_structtype(obj):
    if isinstance(obj, scipy.io.matlab._mio5_params.mat_struct):
        return structtype(**{
            field: matstruct_to_structtype(getattr(obj, field))
            for field in obj._fieldnames
        })

    elif isinstance(obj, dict):
        return {
            key: matstruct_to_structtype(value)
            for key, value in obj.items()
        }

    elif isinstance(obj, (list, tuple)):
        return type(obj)(matstruct_to_structtype(x) for x in obj)

    elif isinstance(obj, numpy.ndarray) and obj.dtype == object:
        return numpy.array(
            [matstruct_to_structtype(x) for x in obj.flat],
            dtype=object
        ).reshape(obj.shape)

    return obj

class structtype(collections.abc.MutableMapping):
    def __init__(self,struct_fields=None,field_values=None,**kwargs):
        if not(type(struct_fields) is type(None)):
            #assert not(type(values) is type(None)),"if you provide field names, you must provide field values"
            if not self._is_iterable(struct_fields):
                struct_fields = [struct_fields]
                field_values = [field_values]
            kwargs.update({f:v for f,v in zip(struct_fields,field_values)})
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
        return self
    def SetAttr(self,field,value):
        if not self._is_iterable(field):
            field = [field]
            value = [value]
        self.__dict__.update({f:v for f,v in zip(field,value)})
        return self
    def GetFields(self,sep='; '):
        return sep.join([ k for k in self.__dict__.keys() if (k[0:2] != '__') and (k[-2:] != '__') ])
        #return self.__dict__.keys()
    def IsField(self,field):
        return field in self.__dict__.keys()
    def RemoveField(self,field):
        return self.__dict__.pop(field,None)
    def RemoveFields(self,*fields):
        r = []
        for k in fields:
            r.append(self.__dict__.pop(k,None))
        return r
    def KeepFields(self,*fields):
        keys = list(self.__dict__.keys())
        for k in keys:
            if not (k in fields):
                self.__dict__.pop(k,None)
    def keys(self):
        return self.__dict__.keys()
    def items(self):
        return self.__dict__.items()
    def values(self):
        return self.__dict__.values()
    def pop(self,key,default_value=None):
        if type(key) is str:
            return self.__dict__.pop(key,default_value)
        elif isinstance(key,collections.abc.Iterable):
            r = []
            for k in key:
                r.append(self.__dict__.pop(k,default_value))
            return r
        else:
            raise ValueError('key must be a string or a list of strings')
    def to_dict(self):
        def convert(obj):
            if isinstance(obj, structtype):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert(v) for v in obj)
            elif obj is None:
                return numpy.array([])
            else:
                return obj
        return convert(self)
    def __setitem__(self,label,value):
        self.__dict__[label] = value
    def __getitem__(self,label):
        return self.__dict__[label]
    def __repr__(self):
        char_lim      = 50
        char_arg_name = 20
        get_repr      = lambda r: r[:char_lim]+'...' if len(r) > char_lim else r
        type_name     = type(self).__name__
        arg_strings   = []
        star_args     = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_name     = (name[:(char_arg_name-3)] + '...') if len(name) > char_arg_name else name.rjust(char_arg_name)
                arg_strings.append('%s: %s' % (arg_name, get_repr(repr(value)).replace('\n','').strip()  ))
            else:
                star_args[name] = get_repr(repr(value))
        if star_args:
            arg_strings.append('**%s' % star_args)
        sep = '\n' #if len(arg_strings) > 3 else '; '
        return '%s(\n%s\n)' % (type_name, sep.join(arg_strings))
    def _get_kwargs(self):
        return sorted(self.__dict__.items())
    def _get_args(self):
        return []
    def _is_iterable(self,obj):
        return (type(obj) is list) or (type(obj) is tuple)
    def __delitem__(self,*args):
        self.__dict__.__delitem__(*args)
    def __len__(self):
        return self.__dict__.__len__()
    def __iter__(self):
        return iter(self.__dict__)