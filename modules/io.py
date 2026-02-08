import os
import re
import copy
import glob
import argparse
import collections.abc
import scipy.io
import numpy
import numpy.core.records
from modules.mouse_track_helper_func_class import structtype

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
    parser.add_argument('-ntrials'       , nargs=1, required=False, metavar='INT'  , type=int    , default=get_param_value('ntrials'       , defaultValues, [10])   , help='Number of trials to repeat the simulation')
    parser.add_argument('-tTrans'        , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('tTrans'        , defaultValues, [50.0]) , help='[[ NOT IMPLEMENTED ]] units: dt. Transient time to discard before measurements')
    parser.add_argument('-tTotal'        , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('tTotal'        , defaultValues, [200.0]), help='units: dt. Total simulation time')
    parser.add_argument('-dt'            , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('dt'            , defaultValues, [0.01]) , help='Integration time step')
    parser.add_argument('-v_tract'       , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('v_tract'       , defaultValues, [5.0])  , help='conduction speed in tract')
    
    parser.add_argument('-Z0'            , nargs=1, required=False, metavar='FLOAT', type=complex, default=get_param_value('Z0'            , defaultValues, [0.0]), help='Mean of the initial condition distribution')
    parser.add_argument('-Z0_std'        , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('Z0_std'        , defaultValues, [0.1]), help='Width of uniform distribution for initial conditions')
    parser.add_argument('-Z_amp_escape'  , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('Z_amp_escape'  , defaultValues, [1.0]), help='Amplitude of Z(t) for measuring escape times')

    parser.add_argument('-alpha'         , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('alpha'         , defaultValues, [0.1])   , help='Noise intensity')
    parser.add_argument('-beta'          , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('beta'          , defaultValues, [0.001]) , help='Global coupling strength')
    parser.add_argument('-omega'         , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('omega'         , defaultValues, [5.0])   , help='Mean natural frequency of oscillators')
    parser.add_argument('-omega_range'   , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('omega_range'   , defaultValues, [0.05])  , help='Width of uniform distribution for natural frequencies')
    parser.add_argument('-lmbda'         , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('lmbda'         , defaultValues, [0.6])   , help='Mean distance to Hopf bifurcation')
    parser.add_argument('-lmbda_range'   , nargs=1, required=False, metavar='FLOAT', type=float  , default=get_param_value('lmbda_range'   , defaultValues, [0.10])  , help='Width of uniform distribution for lmbda')

    # general parameters
    parser.add_argument('-outputFilePrefix'      , nargs=1, required=False, metavar='PREFIX', type=str,   default=get_param_value('outputFilePrefix'       , defaultValues,['osc'])    , help='prefix of the output file name')
    parser.add_argument('-input_type'            , nargs=1, required=False, metavar='STR'   , type=str,   default=get_param_value('input_type'             , defaultValues,['mat'])    , choices=['txt','mat'], help='Input file type: txt or mat')
    parser.add_argument('-input_code'            , nargs=1, required=False, metavar='STR'   , type=str,   default=get_param_value('input_code'             , defaultValues,[''])       , help='code of the individual to simulate')
    parser.add_argument('-input_dir_mat'         , nargs=1, required=False, metavar='STR'   , type=str,   default=get_param_value('input_dir_mat'          , defaultValues,['']   )    , help='(for mat input only) directory where input matrices are stored')
    parser.add_argument('-input_dir_FL'          , nargs=1, required=False, metavar='STR'   , type=str,   default=get_param_value('input_dir_FL'           , defaultValues,['']   )    , help='(for txt input only) directory where input matrices are stored')
    parser.add_argument('-input_dir_FN'          , nargs=1, required=False, metavar='STR'   , type=str,   default=get_param_value('input_dir_FN'           , defaultValues,['']   )    , help='(for txt input only) directory where input matrices are stored')
    parser.add_argument('-input_dir_FMRI'        , nargs=1, required=False, metavar='STR'   , type=str,   default=get_param_value('input_dir_FMRI'         , defaultValues,['']   )    , help='(for txt input only) directory where input matrices are stored')
    parser.add_argument('-input_var_FL'          , nargs=1, required=False, metavar='STR'   , type=str,   default=get_param_value('input_var_FL'           , defaultValues,['M_FL'])   , help='(for mat input only) name of the variable containing the matrix for fiber length in the mat file')
    parser.add_argument('-input_var_FN'          , nargs=1, required=False, metavar='STR'   , type=str,   default=get_param_value('input_var_FN'           , defaultValues,['M_FN'])   , help='(for mat input only) name of the variable containing the matrix for fiber number in the mat file')
    parser.add_argument('-input_var_FMRI'        , nargs=1, required=False, metavar='STR'   , type=str,   default=get_param_value('input_var_FMRI'         , defaultValues,['M_FMRI']) , help='(for mat input only) name of the variable containing the matrix for rs-fMRI correlation matrix in the mat file')

    # simulation flags
    parser.add_argument('-normalizecoupling'     , required=False, action='store_true', default=False, help='If true, divides coupling beta by the number of inputs for each node')
    parser.add_argument('-savealltau'            , required=False, action='store_true', default=False, help='If true, saves all escape times of all trials for all nodes (OUTPUT MEMORY = 2*8*n_subjects*ntrials*N_nodes ~ 300MB; SIMULATION MEMORY = 2*8*n_subjects*(tTotal+max_delay)/dt) ~ 60 MB; for 60 subjects, 1000 trials and 306 nodes;')
    parser.add_argument('-writeOnRun'            , required=False, action='store_true', default=False, help='[[ NOT IMPLEMENTED ]] If true, writes output during run')
    parser.add_argument('-testrun'               , required=False, action='store_true', default=False, help='If true, runs only 1 subject and saves the Z variable for debug purpose (e.g., for checking if network activity diverged)')

    return parser


def get_simParam_struct_validate_args(args):
    s = namespace_to_structtype(args) # fix scalar input parameters automatically in this conversion
    s.ntrials      = input_positive_int(s.ntrials)
    s.tTrans       = input_float_in_range('tTrans'       , s.tTrans      , 0.0)
    s.tTotal       = input_float_in_range('tTotal'       , s.tTotal      , s.tTrans)
    s.dt           = input_float_in_range('dt'           , s.dt          , 0.0)
    s.Z0_std       = input_float_in_range('Z0_std'       , s.Z0_std      , 0.0)
    s.omega_range  = input_float_in_range('omega_range'  , s.omega_range , 0.0)
    s.lmbda_range  = input_float_in_range('lmbda_range'  , s.lmbda_range , 0.0)
    if s.input_type == 'mat':
        assert os.path.isdir(s.input_dir_mat), f'*** invalid input dir: {s.input_dir_mat}'
        assert len(s.input_var_FN)  >0, '*** input_var_FN cannot be empty because input_type==mat'
        assert len(s.input_var_FL)  >0, '*** input_var_FL cannot be empty because input_type==mat'
        assert len(s.input_var_FMRI)>0, '*** input_var_FMRI cannot be empty because input_type==mat'
    else:
        assert os.path.isdir(s.input_dir_rsfMRI)     , f'*** invalid input dir: {s.input_dir_rsfMRI}'
        assert os.path.isdir(s.input_dir_fiberlength), f'*** invalid input dir: {s.input_dir_fiberlength}'
        assert os.path.isdir(s.input_dir_fibernumber), f'*** invalid input dir: {s.input_dir_fibernumber}'
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

def input_int_in_range(argname,value,a,b):
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

def struct_array_for_scipy(field_names,*fields_data):
    """
    returns a data structure which savemat in scipy.io interprets as a MATLAB struct array
    the order of field_names must match the order in which the remaining arguments are passed to this function
    such that
    s(j).(field_names(i)) == fields_data[i][j], identified by field_names[i]

    field_names ->  comma-separated string listing the field names;
                        'field1,field2,...' -> field_names(1) == 'field1', etc...
    fields_data ->  each extra argument entry is a list with the data for each field of the struct
                        fields_data[i][j] :: data for field i in the element j of the struct array: s(j).(field_names(i))
    
    returns
        numpy record array S where
        S[field_names[i]][j] == fields_data[i][j]
    """
    fn_list = field_names.split(',')
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
        match = re.search(r'_(0?\d{3}_\d)(?:_|\.|\b)', s)
        return match.group(1) if match else None
    else:
        return [get_subject_code(ss) for ss in s]

def get_input_txt_file_lists(FL_dir,FN_dir,FMRI_dir):
    file_list_FL   = glob.glob(os.path.join(FL_dir   ,'*.txt'))
    file_list_FN   = glob.glob(os.path.join(FN_dir   ,'*.txt'))
    file_list_FMRI = glob.glob(os.path.join(FMRI_dir ,'*.txt'))

    codes_FL       = set(get_subject_code(get_filename_no_ext(file_list_FL)))
    codes_FN       = set(get_subject_code(get_filename_no_ext(file_list_FN)))
    codes_FMRI     = set(get_subject_code(get_filename_no_ext(file_list_FMRI)))
    codes_valid    = sorted(list(codes_FL & codes_FN & codes_FMRI))
    codes_sort_map = { c:k for k,c in enumerate(codes_valid) }

    file_list_FL_valid   = sorted([ f for f in file_list_FL   if get_subject_code(get_filename_no_ext(f)) in codes_valid ],key=lambda f: codes_sort_map[get_subject_code(get_filename_no_ext(f))])
    file_list_FN_valid   = sorted([ f for f in file_list_FN   if get_subject_code(get_filename_no_ext(f)) in codes_valid ],key=lambda f: codes_sort_map[get_subject_code(get_filename_no_ext(f))])
    file_list_FMRI_valid = sorted([ f for f in file_list_FMRI if get_subject_code(get_filename_no_ext(f)) in codes_valid ],key=lambda f: codes_sort_map[get_subject_code(get_filename_no_ext(f))])
    return codes_valid,file_list_FL_valid,file_list_FN_valid,file_list_FMRI_valid

def get_input_mat_file_list(MAT_dir):
    fl             = glob.glob(os.path.join(MAT_dir,'*.mat'))
    codes          = sorted(get_subject_code(get_filename_no_ext(fl)))
    codes_sort_map = { c:k for k,c in enumerate(codes) }
    return codes, sorted(fl,key=lambda f:codes_sort_map[get_subject_code(get_filename_no_ext(f))])

def get_input_matrices_txt(input_txt_file_FL,input_txt_file_FN,input_txt_file_FMRI):
    return numpy.loadtxt(input_txt_file_FL),numpy.loadtxt(input_txt_file_FN),numpy.loadtxt(input_txt_file_FMRI)

def get_input_matrices_mat(input_mat_file,varname_FL,varname_FN,varname_FMRI):
    d = scipy.io.loadmat(input_mat_file,squeeze_me=True)
    return d[varname_FL],d[varname_FN],d[varname_FMRI]

def select_code(code,file_list,return_as_list=True):
    for f in file_list:
        if code == get_subject_code(get_filename_no_ext(f)):
            return [f] if return_as_list else f
    return None

def select_code_or_die(code,file_list):
    if len(code):
        file_list = select_code(code,file_list,return_as_list=True)
        assert len(file_list), f'*** selected code not found: input_code == {code}'
    return file_list

def load_escape_times_file(fnames):
    d = { k:(v if k=='codes' else _recordarray_to_structtype(v)) for k,v in scipy.io.loadmat(fnames,squeeze_me=True).items() if (k[:2]!='__') and (k[-2:]!='__') }
    return d
