ec50_unit = 1 ## um
import pandas as pd
from threading import Thread,Lock
from apscheduler.schedulers.background import BackgroundScheduler
from multiprocessing import Pool
import time
lock = Lock()


def process_each_drug(ec50, einf, hs, drug):

    for cell in ec50.columns:
            for pert_idose in [0.04, 0.12, 0.37, 1.11, 3.33, 10.0]:
                try:
                    df = pd.DataFrame(columns = ['pert_id', 'cell_id', 'pert_idose', 'ehill'])
                    row_name = drug+'_'+cell+'_'+str(pert_idose)
                    einf_local = einf.loc[drug, cell]
                    ec50_local = ec50.loc[drug, cell]
                    hs_local = hs.loc[drug, cell]
                    ehill = 100 + ( einf_local - 100)/(1+(ec50_local/pert_idose) ** hs_local)
                    df.loc[row_name] = pd.Series({'pert_id': drug, 'cell_id': cell, 'pert_idose': str(pert_idose) + ' um', 'ehill': ehill})
                    print('{0!r} is done'.format(drug))
                except:
                    print("{0!r} and {1!r} are not available".format(drug, cell))
                    continue
    df.to_csv('/workspace/l1000_ph2/PharmacoDB/CTRPv2/ehill/ehill_' + str(drug) + '.csv')

def ehill(ec50, einf, hs):
    try:
        process_pool = Pool(10)
        for drug in ec50.index:
            p.apply_async(process_each_drug, args=(ec50, einf, hs, drug))
        print('Waiting for all subprocesses done...')
        process_pool.close()
        process_pool.join()
        print('All subprocesses done.')
    except:
        raise
    finally:
        process_pool.close()



if __name__ == "__main__":

    ec50 = pd.read_csv('/workspace/l1000_ph2/PharmacoDB/CTRPv2/ec50_ctrpv2.csv', index_col = 0)
    einf = pd.read_csv('/workspace/l1000_ph2/PharmacoDB/CTRPv2/einf_ctrpv2.csv', index_col = 0)
    hs = pd.read_csv('/workspace/l1000_ph2/PharmacoDB/CTRPv2/hs_ctrpv2.csv', index_col = 0)

    ehill(ec50, einf, hs)

    # ehill_df = ehill(ec50, einf, hs)
    # ehill_df.to_csv('/workspace/l1000_ph2/PharmacoDB/CTRPv2/ehill/ehill_df.csv')