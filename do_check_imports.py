#!/usr/bin/env python

# This is a quick program to help check the importability and version
# numbers of modules used in the BSHE/SIMBA project.  This script
# should just be runnable from the same directory where any
# script/notebook with a SIMBA model to run exists.
#
# This script creates a log file of the current Python version and
# whether a module could be imported, and then if it could be, what
# version number it has.
#
# Surprisingly, some common modules don't have version numbers to
# report, as far as I can tell.  These modules list will likely list
# 'unknown' as the version number:
#   model (OK, this one is in the SIMBA repo), multiprocessing, pickle, time
# ... and this one lists 'editable' for some reason:
#   fastkde
# 
# auth: PA Taylor (SSCC, NIMH, NIH, USA)
# ver : 1.0 (2025-09-13)
#
# ============================================================================

import sys

# all python modules in current BSHE repo: keep this uptodate as they change
all_mod  = [ 'fastkde', 
             'matplotlib', 'model', 'multiprocessing', 
             'numpy', 'nibabel', 'nilearn', 
             'pandas', 'pickle', 
             'sklearn', 
             'time', 'torch', 'tqdm' ]

# name of the log file to write
logfile  = 'log_python_mods.txt'

# ============================================================================

def import_module_and_get_ver(mod):
    """Try to import the string name of the module mod.  

If the import is successful, return the imported module obj, and
report its version number.

If the import is not successful, return None and a string message
    """

    try:
        imp_mod = __import__(mod)
        ver     = None
    except:
        imp_mod = None
        ver     = 'failed_import'

    if ver is None:
        try: 
            ver = imp_mod.__version__
        except:
            ver = 'unknown'

    return imp_mod, ver


# ============================================================================

if __name__ == "__main__" : 

    # prep to loop
    nmod     = len(all_mod)
    all_text = []

    # title line of table
    sss = "{mod:20s} : {ver:>15s}".format(mod="# Module", ver="Version")
    all_text.append(sss)
    sss = "{mod:20s}   {ver:>15s}".format(mod="# "+"-"*18, ver="-"*15)
    all_text.append(sss)
    
    # get current Python version info
    mod = 'python'
    ver = sys.version.split()[0]
    sss = "{mod:20s} : {ver:>15s}".format(mod=mod, ver=ver)
    all_text.append(sss)

    for mm in range(nmod):
        mod = all_mod[mm]

        # do the main work of importing and checking version
        imp_mod, ver = import_module_and_get_ver(mod)

        sss = "{mod:20s} : {ver:>15s}".format(mod=mod, ver=ver)
        all_text.append( sss )
        print("  ", sss)

    # write out log file of findings
    fff = open(logfile, 'w')
    for text in all_text :
        fff.write(text + '\n')
    fff.close()

    # fare thee well
    sys.exit(0)

