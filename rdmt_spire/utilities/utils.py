import re
import time
from typing import List

from ..constants.dmd import FileTypes


class Timer:
    """
    Convenient class for computing running time of different parts of the code. 
    
    It can be used to track the timing of different executions to help understand the resources 
    each monitor will require.
    """
    def __init__(self):
        self.reset()
        self.result: List[str]=[]

    def reset(self):
        self.start=time.time()

    def log(self,line_number:int):
        self.result.append(f"{line_number:<10d} {self.get():8.6f}")

    def print_log(self):
        print(f"{'Line #':10s} {'Time [s]':8s}")
        print('---------------------')
        for s in self.result:
            print(s)
        print('---------------------')

    def get(self, reset:bool=False) -> float:
        self.end=time.time()
        if reset:
            self.reset()
        return (self.end-self.start) 
    
    def print(self, reset:bool=False):
        print('Time:', self.get(reset=reset))


def get_info_from_filename(filename, file_type):
    """ 
    Parse a WFI filename and extract observation metadata.

    This function validates the input filename against the WFI filename
    conventions and, when matched, parses out key observation fields such as
    program, execution, pass, segment, observation, visit, and exposure numbers,
    as well as the detector and optical element.

    For detailed definitions of the filename components, see the RDox page
    on filename conventions:
    https://roman-docs.stsci.edu/data-handbook/wfi-data-levels-and-products#WFIDataLevelsandProducts-Table-L1L2-Components

    Parameters
    ----------
    filename : str
        The name of the file to parse (e.g.,
        ``r123450102003004005006_0001_WFI10_F158_cal.asdf``).
    file_type : FileTypes
        Enumeration value indicating the file type. Currently only
        ``FileTypes.L2_SCIENCE`` is supported.

    Returns
    -------
    dict
        A dictionary with parsed observation fields:

        - ``program_num`` : int
            5-digit program number (PPPPP).
        - ``execution_num`` : int
            2-digit execution number (CC).
        - ``pass_num`` : int
            3-digit pass number (AAA).
        - ``segment_num`` : int
            3-digit segment number (SSS).
        - ``obs_num`` : int
            3-digit observation number (OOO).
        - ``visit_num`` : int
            3-digit visit number (VVV).
        - ``exposure_num`` : int
            4-digit exposure number (eeee).
        - ``visit_id`` : str
            Concatenation ``PPPPPCCAAASSSOOOVVV`` (characters 1–20 of the basename).
        - ``detector`` : str
            Detector identifier (e.g., ``WFI01``).
        - ``optical_element`` : str
            Optical element token (e.g., ``F158``, ``GRISM``, ``PRISM``).

    """
    if file_type == FileTypes.L2_SCIENCE:
        filename_pattern = r'r\d{19}_\d{4}_wfi\d{2}_(f\d{3}|(g|p)rism)_cal.asdf'
        if re.match(filename_pattern, filename.lower()):
            split_str = filename.split('_')
            det_str = [x.upper() for x in split_str if x[0].upper() == 'W'][0]
            opt_elem_str = [x.upper() for x in split_str if x[0].upper() in ['F', 'P', 'G']][0]
            obs_nums = {
                'program_num': int(filename[1:6]), #PPPPP
                'execution_num': int(filename[6:8]), #CC
                'pass_num': int(filename[8:11]), #AAA
                'segment_num': int(filename[11:14]), #SSS
                'obs_num': int(filename[14:17]), #OOO
                'visit_num': int(filename[17:20]), #VVV
                'exposure_num': int(filename[21:25]), #eeee
                'visit_id': filename[1:20], #PPPPPCCAAASSSOOOVVV
                'detector': det_str, # WFIXX
                'optical_element': opt_elem_str, # FXXX, GRISM, PRISM
            }
        else:
            raise ValueError(f"Filename does not match expected structure. Received: {filename}.")
    elif file_type == FileTypes.L1_GUIDE_WINDOW:
        filename_pattern = r'r\d{19}_\d{1}_wfi\d{2}_(f\d{3}|(g|p)rism|dark|none)_gw.asdf'
        if re.match(filename_pattern, filename.lower()):
            split_str = filename.split('_')
            det_str = [x.upper() for x in split_str if x[0].upper() == 'W'][0]
            opt_elem_str = [x.upper() for x in split_str if x[0].upper() in ['F', 'P', 'G', 'N']][0]
            obs_nums = {
                'program_num': int(filename[1:6]), #PPPPP
                'execution_num': int(filename[6:8]), #CC
                'pass_num': int(filename[8:11]), #AAA
                'segment_num': int(filename[11:14]), #SSS
                'obs_num': int(filename[14:17]), #OOO
                'visit_num': int(filename[17:20]), #VVV
                'gw_acquisition_num': int(filename[21:22]), #a
                'visit_id': filename[1:20], #PPPPPCCAAASSSOOOVVV
                'detector': det_str, # WFIXX
                'optical_element': opt_elem_str, # FXXX, GRISM, PRISM, DARK, NONE
            }
        else:
            raise ValueError(f"Filename does not match expected structure. Received: {filename}.")
    else:
        raise NotImplementedError(f"File tyle {file_type} is not yet implemented in get_info_from_filename")
    
    return obs_nums
    