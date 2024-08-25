from sklearn.neighbors import KNeighborsClassifier

def get_model():
    # Directly instantiate the model with the best parameters
    best_knn = KNeighborsClassifier(weights='distance', p=1, n_jobs=-1)
    
    return best_knn
