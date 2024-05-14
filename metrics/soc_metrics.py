import os
import sys
import cv2
from tqdm import tqdm
import metrics as M
import json
import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from hce_metric_main_refine import once_compute_hec
from miou import measure_pa_miou,Evaluator

_EPS = 1e-16
_TYPE = np.float64

def get_files(path,name='.pkl'):
    file_lan = []
    for filepath,dirnames,filenames in os.walk(path):
        for filename in filenames:
            if name not in os.path.join(filepath,filename):
                file_lan.append(os.path.join(filepath,filename))
    return file_lan

def once_compute(gt_root,gt_name,pred_root,FM,WFM,SM,EM,MAE,Evaltor):
    gt_path = os.path.join(gt_root, gt_name)
    pred_path = os.path.join(pred_root, gt_name)
    # print(gt_path,pred_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    fm = FM.step(pred=pred, gt=gt)
    wfm = WFM.step(pred=pred, gt=gt)
    mae = MAE.step(pred=pred, gt=gt)
    sm = SM.step(pred=pred, gt=gt)
    em = EM.step(pred=pred, gt=gt)
    if pred.sum() == 0:
        miou = 0
        mbiou = 0
    else:
        Evaltor.add_batch(gt, pred)
        miou = Evaltor.Mean_Intersection_over_Union()
        Evaltor.add_b_batch(gt, pred)
        mbiou = Evaltor.Mean_Intersection_over_Union()
    FP_points, FP_indep, FN_points, FN_indep = once_compute_hec(gt_root,gt_name,pred_root,'')
    hce = FP_points+FP_indep+FN_points+FN_indep
    return {'fm':fm,
            'wfm':wfm,
            # 'wfm':0,
            'mae':mae,
            # 'mae':0,
            'sm':sm,
            # 'sm':0,
            'em':em,
            # 'em':0,
            'hce':hce,
            # 'hce':0,
            'miou':miou,
            # 'miou':0,
            'mbiou':mbiou,
            # 'mbiou':0
            }

def main():
    args = parser.parse_args()
    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()
    Evaltor = Evaluator(2)

#<<<<<<< HEAD:codes/soc_eval.py

    gt_roots = [r'/metrics/valid_gt',]
    
    pred_roots = [r'/metrics/valid',]
    
    allfile = pd.DataFrame()
    for i in range(gt_roots.__len__()):
        gt_root = gt_roots[i]
        pred_root = pred_roots[i]
        gt_name_list = get_files(pred_root)
        gt_name_list = sorted([x.split('/')[-1] for x in gt_name_list])
        # print(gt_root,pred_root)
        # for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        results = Parallel(n_jobs=8)(delayed(once_compute)(gt_root,gt_name,pred_root,FM,WFM,SM,EM,MAE,Evaltor) for gt_name in tqdm(gt_name_list, total=len(gt_name_list)))
        fm,wfm,sm,em,mae,hce,ap,miou,apb,mbiou = [],[],[],[],[],[],[],[],[],[]
        for result in results:
            fm.append([result['fm']])
            wfm.append([result['wfm']])
            mae.append([result['mae']]) 
            sm.append([result['sm']])
            em.append([result['em']])
            hce.append([result['hce']])
            miou.append([result['miou']])
            mbiou.append([result['mbiou']])
            
        # print(np.array(fm, dtype=_TYPE).shape)
        fm = np.mean(np.array(fm, dtype=_TYPE), axis=0)
        # print(fm.shape)
        wfm = np.mean(np.array(wfm, dtype=_TYPE))
        mae = np.mean(np.array(mae, dtype=_TYPE))
        sm = np.mean(np.array(sm, dtype=_TYPE))
        em = np.mean(np.array(em, dtype=_TYPE), axis=0)
        hce = np.mean(np.array(hce, dtype=_TYPE))
        miou = np.mean(np.array(miou, dtype=_TYPE))
        mbiou = np.mean(np.array(mbiou, dtype=_TYPE))
        onefile = pd.DataFrame()
        results = {'maxFm':fm.max(),
            'wFmeasure':wfm,
            'MAE':mae, 
            'Smeasure:':sm, 
            'meanEm':em.mean(),
            'hce':hce,
            'miou':miou,
            'mbiou':mbiou}
        results = pd.DataFrame.from_dict([results]).T
        onefile = pd.concat([onefile,results])
        print(
            'Method:', args.method+str(i), ', ',
            'maxFm:', fm.max().round(3),'; ',
            'wFmeasure:', wfm.round(3), '; ',
            'MAE:', mae.round(3), '; ',
            'Smeasure:', sm.round(3), '; ',
            'meanEm:', em.mean().round(3), '; ',
            'hce:',hce.round(3), '; ',
            'miou:',miou.round(3), '; ',
            'mbiou:',mbiou.round(3), '; ',
            sep=' '
        )
        # onefile.to_csv(args.method+str(i)+".csv")
        allfile = pd.concat([allfile.T,onefile.T]).T
    allfile.to_csv(args.method+"all.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='3-1024-locs-')
    main()