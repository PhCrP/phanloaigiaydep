import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sbn
from sklearn.tree import plot_tree


def mhCart(cart_model):
    
    plt.figure(figsize=(13, 10))
    plot_tree(cart_model, filled=True, rounded=True, feature_names=None, class_names=True)
    plt.show()


def mtnl(y_test, y_pred):
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sbn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("Ma trận nhầm lẫn")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.show()


def accPrdRec(y_test, y_pred):
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Độ chính xác:", accuracy)

    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision:", precision)

    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall:", recall)
    
    metrics = ['Accuracy', 'Precision', 'Recall']
    scores = [accuracy * 100, precision * 100,
              recall * 100]  

    plt.figure(figsize=(6, 8))
    plt.bar(metrics, scores, color=['green', 'orange', 'red'])

    plt.title('Số liệu đánh giá mô hình')
    plt.ylabel('Scores (%)')
    plt.ylim([80, 100]) 

    plt.yticks(np.arange(80, 100.5, 0.5))

    for i, v in enumerate(scores):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.show()
