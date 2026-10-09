import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, cohen_kappa_score

def evaluate_predictions(real_aux, labels, num_classes):
    output = np.argmax(real_aux.data.cpu().numpy(), axis=1)
    answer = labels.data.cpu().numpy()
    accuracy = accuracy_score(answer, output)
    f1 = f1_score(answer, output, average='weighted')
    report = classification_report(answer, output, zero_division=0)
    conf_matrix = confusion_matrix(answer, output, labels=list(range(num_classes+1)))
    return accuracy, f1, report, conf_matrix

def evaluate_predictions2(real_aux, labels, num_classes):
    output = np.argmax(real_aux.data.cpu().numpy(), axis=1)
    sorted_indices = np.argsort(real_aux.cpu().numpy(), axis=1)[:, ::-1]
    second_max_indices = sorted_indices[:, 1]
    output = np.where(output==num_classes, second_max_indices, output)
    answer = labels.data.cpu().numpy()
    accuracy = accuracy_score(answer, output)
    f1 = f1_score(answer, output, average='weighted')
    qwk = cohen_kappa_score(answer, output, weights='quadratic')
    report = classification_report(answer, output, zero_division=0)
    conf_matrix = confusion_matrix(answer, output, labels=list(range(num_classes)))
    return accuracy, f1, qwk, report, conf_matrix, output

def get_qwk(output, answer):
    score = cohen_kappa_score(answer, output, weights='quadratic')
    return score
