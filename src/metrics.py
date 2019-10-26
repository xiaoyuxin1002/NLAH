import warnings
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def performance_classification(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    accuracy = accuracy_score(labels, outputs)
    fscore = f1_score(labels, outputs, average='macro')
    return (accuracy, fscore)