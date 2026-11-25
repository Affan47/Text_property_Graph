"""Regression and explicitly labelled EPSS-category diagnostics, not exploitation metrics."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .protocol import require


def score_predictions(actual, predicted, thresholds=(.01,.1)):
    y,p = np.asarray(actual,dtype=float),np.asarray(predicted,dtype=float)
    require(y.ndim==p.ndim==1 and y.shape==p.shape and len(y)>0
            and np.isfinite(y).all() and np.isfinite(p).all(),'Require aligned finite observed predictions')
    require(((0<=y)&(y<=1)).all() and ((0<=p)&(p<=1)).all(),'Scores must be in [0,1]')
    error=p-y; ae=np.abs(error);total=float(((y-y.mean())**2).sum())
    out={'n':len(y),'mae':float(ae.mean()),'rmse':float(np.sqrt((error**2).mean())),
         'median_absolute_error':float(np.median(ae)),'mean_signed_error':float(error.mean()),
         'q90_absolute_error':float(np.quantile(ae,.90)),'q95_absolute_error':float(np.quantile(ae,.95)),
         'r2':float(1-(error**2).sum()/total) if total>0 else None,
         'spearman':float(spearmanr(y,p).statistic) if len(set(y))>1 and len(set(p))>1 else None,
         'undefined_reasons':{},'threshold_categories':{}}
    if out['r2'] is None:out['undefined_reasons']['r2']='constant_actual_scores'
    if out['spearman'] is None:out['undefined_reasons']['spearman']='constant_actual_or_predicted_scores'
    for t in thresholds:
        require(0<t<1,'Invalid category threshold')
        yes=y>=t;guess=p>=t
        tp,tn,fp,fn=[int(v.sum()) for v in (yes&guess,~yes&~guess,~yes&guess,yes&~guess)]
        precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
        specificity=tn/(tn+fp) if tn+fp else None
        two_classes=bool(yes.any() and (~yes).any())
        out['threshold_categories'][str(t)]={'meaning':f'EPSS >= {t}, not exploitation',
            'confusion_matrix_true_rows_predicted_columns':[[tn,fp],[fn,tp]],
            'accuracy':(tp+tn)/len(y),'precision':precision,'recall':recall,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'balanced_accuracy':(recall+specificity)/2 if two_classes else None,
            'roc_auc':float(roc_auc_score(yes,p)) if two_classes else None,
            'average_precision':float(average_precision_score(yes,p)) if two_classes else None,
            'undefined_reasons':{'precision':'no_predicted_positives' if precision is None else None,
                'recall':'no_actual_positives' if recall is None else None,
                'f1':'no_actual_or_predicted_positives' if 2*tp+fp+fn==0 else None,
                'balanced_accuracy':'single_actual_class' if not two_classes else None,
                'rank_metrics':'single_actual_class' if not two_classes else None}}
    return out


def balanced_forecast_mae(rows):
    """Equal CVEs, equal eligible anchors within CVE, equal observed leads within anchor."""
    groups={};seen=set()
    for r in rows:
        key=(r['cve'],r['anchor'],r['lead'])
        require(key not in seen,'Duplicate forecast coordinate');seen.add(key)
        if r['actual'] is None:continue
        score_predictions([r['actual']],[r['prediction']],thresholds=())
        groups.setdefault(r['cve'],{}).setdefault(r['anchor'],[]).append(abs(r['prediction']-r['actual']))
    require(bool(groups),'No observed targets')
    return float(np.mean([np.mean([np.mean(v) for v in anchors.values()]) for anchors in groups.values()]))
