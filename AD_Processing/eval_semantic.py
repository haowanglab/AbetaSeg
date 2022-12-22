from medpy import metric
import os
import tifffile
import numpy as np

def calculate_metric_percase(pred, gt):
    """
        Calculate metric per image.
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jc = metric.binary.jc(pred, gt)
        rec = metric.binary.recall(pred, gt)
        pre = metric.binary.precision(pred, gt)
        sst = metric.binary.sensitivity(pred, gt)
        return dice, hd95, jc, rec, pre, sst
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0, 0, 0, 0, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0

def evaluate_sem(pred_root, Region_gt_root):
    """
        Evaluate the semantic segmentation result with Ground Truth.
    """
    gt_names = os.listdir(Region_gt_root)
    sem_result = os.path.join(pred_root, '..', 'sem_result')
    if not os.path.exists(sem_result):
        os.mkdir(sem_result)
    result_txt = os.path.join(sem_result,Region_gt_root.split('\\')[-1]+'_sem.txt')
    print(result_txt)
    f = open(result_txt,'w+')
    result_avg = np.zeros(6).astype('float64')
    total = 0
    for gt_name in gt_names:
        print(gt_name)
        gt_path = os.path.join(Region_gt_root, gt_name)
        pred_path = os.path.join(pred_root,gt_name)
        pred = tifffile.imread(pred_path)
        gt = tifffile.imread(gt_path)
        dice, hd95, jc, rec, pre, sst = calculate_metric_percase(pred, gt)
        result_avg += np.array([dice, hd95, jc, rec, pre, sst])
        total+=1
        log_item = gt_name + '\tdice:'+str(dice)+'\thd95:'+ str(hd95)+'\tjaccard:'+ str(jc)+\
              '\trecall:'+ str(rec)+'\tprecision:'+ str(pre)+'\tsensitivity:'+ str(sst)+'\n'
        print(log_item)
        f.writelines(log_item)
    result_avg /= total
    log_item = 'avg:' + '\tdice:'+str(result_avg[0])+'\thd95:'+ str(result_avg[1])+'\tjaccard:'+ str(result_avg[2])+\
              '\trecall:'+ str(result_avg[3])+'\tprecision:'+ str(result_avg[4])+'\tsensitivity:'+ str(result_avg[5])+'\n'
    print(result_avg)
    f.writelines(log_item)
    f.close()


def evalNetwork(Root_list, gt_root):
    """
        Evaluate the network prediction in different brain region.
    """
    for Root in Root_list:
        pred_root = os.path.join(Root, 'model_step2999')
        Region_gt_root = os.path.join(gt_root, 'Cortex')
        evaluate_sem(pred_root, Region_gt_root)
        Region_gt_root = os.path.join(gt_root, 'Hippo')
        evaluate_sem(pred_root, Region_gt_root)
        Region_gt_root = os.path.join(gt_root, 'Other')
        evaluate_sem(pred_root, Region_gt_root)



if __name__ == '__main__':
    Root_list = ['', '', '']
    gt_root = ''
    evalNetwork(Root_list, gt_root)
