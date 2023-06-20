import torch
from sklearn.metrics import roc_auc_score

def accuracy(y_matrix_true, y_matrix_pred, calc_roc=False, single_class=False):
    # if calc_ROC is True, it will instead compute the ROC AUC score (this should be set to True for binary classification)
    if calc_roc:
        yp_tmp = y_matrix_pred.flatten().cpu().detach().numpy()
        #yp_tmp -= yp_tmp.mean()
        #yp_tmp /= yp_tmp.std()
        return None, None, roc_auc_score(y_matrix_true.flatten().cpu().detach().numpy(), yp_tmp)
    if not single_class:
        y_true_class = torch.argmax(y_matrix_true, dim=1)
        y_pred_class = torch.argmax(y_matrix_pred, dim=1)
    else:
        y_true_class = y_matrix_true.flatten()
        y_pred_class = torch.round(y_matrix_pred.flatten())
    assert len(y_true_class) == len(y_pred_class)
    acc = len(torch.where(y_true_class == y_pred_class)[0]) / len(y_true_class)
    return y_true_class, y_pred_class, acc

