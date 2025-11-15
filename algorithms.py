import pandas as pd
import numpy as np

class XGBoostClassifier:

    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3,reg_lambda=1, gamma=0,colsample_bytree=1.0, min_child_weight=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.trees = [] # Will be a list of lists
        self.base_score = None # Will be an array, one score per class
        self.classes_ = None # Stores the unique class labels
        self.n_classes = None # Number of classes

    class XGBoostTree:
        def __init__(self, max_depth=3, reg_lambda=1.0, gamma=0.0, min_child_weight=1):
            self.max_depth = max_depth
            self.reg_lambda = reg_lambda
            self.gamma = gamma
            self.min_child_weight = min_child_weight
            self.tree_dict = None

        def fit(self, X, g, h, depth=0, feature_map=None):
            G = np.sum(g)
            H = np.sum(h)

            n_samples, n_features = X.shape

            if feature_map is None:
                feature_map = np.arange(n_features)

            if depth >= self.max_depth or n_samples <= 1 or H < self.min_child_weight:
                leaf_weight = -G / (H + self.reg_lambda)
                return {"leaf": leaf_weight}

            best_gain = 0.0
            best_split = None

            for feature_index in range(n_features):
                thresholds = X[:, feature_index]
                ordered = np.argsort(thresholds)
                Xj = thresholds[ordered]
                gj = g[ordered]
                hj = h[ordered]
                g_cumsum = np.cumsum(gj)
                h_cumsum = np.cumsum(hj)

                for i in range(1, n_samples):
                    if Xj[i] == Xj[i - 1]:
                        continue

                    G_L, H_L = g_cumsum[i - 1], h_cumsum[i - 1]
                    G_R, H_R = G - G_L, H - H_L

                    if H_L < self.min_child_weight or H_R < self.min_child_weight:
                        continue

                    gain = 0.5 * (
                        (G_L ** 2) / (H_L + self.reg_lambda)
                        + (G_R ** 2) / (H_R + self.reg_lambda)
                        - (G ** 2) / (H + self.reg_lambda)
                    ) - self.gamma

                    if gain > best_gain:
                        best_gain = gain
                        threshold_val = (Xj[i - 1] + Xj[i]) / 2.0
                        left_mask = X[:, feature_index] <= threshold_val
                        right_mask = ~left_mask

                        original_feature_index = feature_map[feature_index]
                        best_split = (original_feature_index, threshold_val, left_mask, right_mask)

            if best_split is None or best_gain <= 0:
                leaf_weight = -G / (H + self.reg_lambda)
                return {"leaf": leaf_weight}

            feature_index, t, left_mask, right_mask = best_split

            left_subtree = self.fit(X[left_mask], g[left_mask], h[left_mask], depth + 1, feature_map)
            right_subtree = self.fit(X[right_mask], g[right_mask], h[right_mask], depth + 1, feature_map)

            return {
                "feature": feature_index,
                "threshold": t,
                "left": left_subtree,
                "right": right_subtree
            }

        def _predict_one(self, x, node):
            # Traverse the tree to get prediction for one sample.
            if "leaf" in node:
                return node["leaf"]

            if x[node["feature"]] <= node["threshold"]:
                return self._predict_one(x, node["left"])
            else:
                return self._predict_one(x, node["right"])

        def predict(self, X):
            return np.array([self._predict_one(x, self.tree_dict) for x in X])

    def _softmax(self, x):
        # Subtracting max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        eps = 1e-15
        n_samples, n_features = X.shape

        # Determine number of classes and store class labels
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        class_map = {label: i for i, label in enumerate(self.classes_)}
        y_mapped = np.array([class_map[label] for label in y])

        y_one_hot = np.zeros((n_samples, self.n_classes))
        y_one_hot[np.arange(n_samples), y_mapped] = 1

        # Initialize base score (initial logits)
        class_counts = np.sum(y_one_hot, axis=0)
        class_probs = class_counts / n_samples
        self.base_score = np.log(np.clip(class_probs, eps, 1 - eps))

        # Initialize predictions (logits)
        y_pred = np.tile(self.base_score, (n_samples, 1))

        for _ in range(self.n_estimators):
            p = self._softmax(y_pred)

            g = p - y_one_hot

            h = p * (1 - p)

            # Sample features for this round
            n_cols_to_sample = int(n_features * self.colsample_bytree)
            feature_indices = np.random.choice(n_features, n_cols_to_sample, replace=False)
            X_sampled_cols = X[:, feature_indices]

            round_trees = []

            # Build one tree for each class
            for k in range(self.n_classes):
                tree = self.XGBoostTree(
                    self.max_depth,
                    self.reg_lambda,
                    self.gamma,
                    self.min_child_weight
                )

                g_k = g[:, k]
                h_k = h[:, k]

                tree.tree_dict = tree.fit(X_sampled_cols, g_k, h_k, feature_map=feature_indices)
                round_trees.append(tree)

                y_pred[:, k] += self.learning_rate * tree.predict(X)

            self.trees.append(round_trees)

    def predict_proba(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]

        # Initialize predictions (logits) with the base score
        pred = np.tile(self.base_score, (n_samples, 1))

        # Sum predictions from all trees
        for round_trees in self.trees:
            for k in range(self.n_classes):
                tree = round_trees[k]
                pred[:, k] += self.learning_rate * tree.predict(X)

        return self._softmax(pred)

    def predict(self, X):
        proba = self.predict_proba(X)

        indices = np.argmax(proba, axis=1)

        return self.classes_[indices]
    
# --- Helper Functions ---
def sigmoid(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def to_one_hot(y, n_classes=None):
    if n_classes is None:
        n_classes = np.max(y) + 1
    one_hot = np.zeros((y.shape[0], n_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.theta_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Convert y to one-hot encoding
        y_one_hot = to_one_hot(y, n_classes)
        
        # Add intercept term
        X_b = np.c_[np.ones((n_samples, 1)), X]
        
        # Initialize weights
        self.theta_ = np.zeros((n_features + 1, n_classes))
        
        for epoch in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_b[indices]
            y_shuffled = y_one_hot[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Compute predictions
                linear = np.dot(X_batch, self.theta_)
                predictions = softmax(linear)
                
                # Compute gradients
                errors = predictions - y_batch
                gradients = (1.0 / len(X_batch)) * np.dot(X_batch.T, errors)
                
                # Update weights
                self.theta_ -= self.learning_rate * gradients
        
        return self

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        linear = np.dot(X_b, self.theta_)
        return softmax(linear)

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

class MultiClassSVM:
    def __init__(self, learning_rate=0.001, lambda_p=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_p = lambda_p
        self.n_iters = n_iters
        self.models_ = []  # Will store (w, b) tuples for each class
        self.classes_ = None

    def _sigmoid(self, x):
        x = np.array(x, dtype=float)
        
        result = np.empty_like(x)
        
        positive_mask = (x >= 0)
        result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
        
        negative_mask = (x < 0)
        exp_x = np.exp(x[negative_mask])
        result[negative_mask] = exp_x / (1 + exp_x)
        
        return result

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        X = np.array(X, dtype=float)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.models_ = []  

        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            
            w = np.zeros(n_features)
            b = 0
            
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    margin = y_binary[idx] * (np.dot(x_i, w) + b)
                    
                    if margin < 1:
                        w -= self.learning_rate * (self.lambda_p * w - y_binary[idx] * x_i)
                        b -= self.learning_rate * (-y_binary[idx])
                    else:
                        w -= self.learning_rate * (self.lambda_p * w)
            
            self.models_.append((w, b))
            
        return self

    def decision_function(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X, dtype=float)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))
        
        for idx, (w, b) in enumerate(self.models_):
            scores[:, idx] = np.dot(X, w) + b
            
        return scores

    def predict_proba(self, X):
        # Get raw decision scores
        scores = self.decision_function(X)
        sig_scores = self._sigmoid(scores)
        
        exp_probas = np.exp(sig_scores - np.max(sig_scores, axis=1, keepdims=True))
        probas = exp_probas / np.sum(exp_probas, axis=1, keepdims=True)
        
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    
#Custom Standard Scaler
class MyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1e-8
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

#Custom Pipeline
class MyPipeline:
    def __init__(self, steps):
        self.steps = steps

    def _get_estimator(self):
        return self.steps[-1][1]

    def fit(self, X, y=None):
        X_transformed = X
        for name, transformer in self.steps[:-1]:
            X_transformed = transformer.fit_transform(X_transformed, y)
        
        estimator = self._get_estimator()
        estimator.fit(X_transformed, y)
        return self
    
    def predict(self, X):
        X_transformed = X
        for name, transformer in self.steps[:-1]:
            X_transformed = transformer.transform(X_transformed)
        
        return self._get_estimator().predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = X
        for name, transformer in self.steps[:-1]:
            X_transformed = transformer.transform(X_transformed)
        
        return self._get_estimator().predict_proba(X_transformed)

#Simple K-Fold Split
class SimpleKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            yield train_indices, val_indices

#Stacking Classifier
class MyStackingClassifier:
    def __init__(self, base_models, meta_model, k=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.k = k
        self.meta_scaler = MyStandardScaler()
        self.retrained_base_models = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        M = len(self.base_models)
        
        #Out-of-Fold predictions for training the meta-model
        Z_oof = np.zeros((n_samples, M * n_classes))
        
        kf = SimpleKFold(n_splits=self.k, shuffle=True, random_state=42)
        
        print("Generating OOF predictions")
        for j, base_model in enumerate(self.base_models):
            print(f"Training base model {j+1}/{M}")
            
            # Create a fresh instance for each base model
            model_copy = base_model.__class__(**base_model.__dict__)
            
            for train_idx, val_idx in kf.split(X, y):
                X_train_fold, y_train_fold = X[train_idx], y[train_idx]
                X_val_fold = X[val_idx]
        
                model_copy.fit(X_train_fold, y_train_fold)
                
                proba = model_copy.predict_proba(X_val_fold)
                Z_oof[val_idx, j*n_classes:(j+1)*n_classes] = proba
        
        print('Training meta model')
        self.meta_scaler.fit(Z_oof)
        Z_oof_scaled = self.meta_scaler.transform(Z_oof)
        self.meta_model.fit(Z_oof_scaled, y)
        
        print('Retraining base model on full train data')
        self.retrained_base_models = []
        for model in self.base_models:
            model_copy = model.__class__(**model.__dict__)
            model_copy.fit(X, y)
            self.retrained_base_models.append(model_copy)
            
        print("Stacking model fitting complete")
        return self

    def predict(self, X):
        n_classes = len(self.classes_)
        M = len(self.retrained_base_models)
        Z_test = np.zeros((X.shape[0], M * n_classes))
        
        for j, model in enumerate(self.retrained_base_models):
            proba = model.predict_proba(X)
            Z_test[:, j*n_classes:(j+1)*n_classes] = proba
            
        Z_test_scaled = self.meta_scaler.transform(Z_test)
        return self.meta_model.predict(Z_test_scaled)

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        M = len(self.retrained_base_models)
        Z_test = np.zeros((X.shape[0], M * n_classes))
        
        for j, model in enumerate(self.retrained_base_models):
            proba = model.predict_proba(X)
            Z_test[:, j*n_classes:(j+1)*n_classes] = proba
            
        Z_test_scaled = self.meta_scaler.transform(Z_test)
        return self.meta_model.predict_proba(Z_test_scaled)

#Inference Model
class StackingInferenceModel:
    def __init__(self, stacking_model):
        self.retrained_base_models = stacking_model.retrained_base_models
        self.meta_model = stacking_model.meta_model
        self.meta_scaler = stacking_model.meta_scaler
        self.classes_ = stacking_model.classes_
    
    def predict(self, X_new):
        n_classes = len(self.classes_)
        M = len(self.retrained_base_models)
        Z_new = np.zeros((X_new.shape[0], M * n_classes))
        
        for j, model in enumerate(self.retrained_base_models):
            proba = model.predict_proba(X_new)
            Z_new[:, j*n_classes:(j+1)*n_classes] = proba
            
        Z_new_scaled = self.meta_scaler.transform(Z_new)
        return self.meta_model.predict(Z_new_scaled)
    
    def predict_proba(self, X_new):
        n_classes = len(self.classes_)
        M = len(self.retrained_base_models)
        Z_new = np.zeros((X_new.shape[0], M * n_classes))
        
        for j, model in enumerate(self.retrained_base_models):
            proba = model.predict_proba(X_new)
            Z_new[:, j*n_classes:(j+1)*n_classes] = proba
            
        Z_new_scaled = self.meta_scaler.transform(Z_new)
        return self.meta_model.predict_proba(Z_new_scaled)
    
class MyKNN:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights  # 'uniform' or 'distance'
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self
    
    def _compute_distances(self, X):
        X = np.array(X)
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2))
        return distances
    
    def predict_proba(self, X):
        distances = self._compute_distances(X)
        
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))
        
        if self.weights == 'uniform':
            for i in range(n_samples):
                for j, cls in enumerate(self.classes_):
                    probas[i, j] = np.sum(nearest_labels[i] == cls) / self.n_neighbors
                    
        elif self.weights == 'distance':
            # Distance-based weighting
            for i in range(n_samples):
                sample_distances = distances[i, nearest_indices[i]]
                
                weights = 1.0 / (sample_distances + 1e-8)
                
                for j, cls in enumerate(self.classes_):
                    mask = (nearest_labels[i] == cls)
                    probas[i, j] = np.sum(weights[mask]) / np.sum(weights)
        
        return probas
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
class PCA:
    """Principal Component Analysis from scratch"""

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X, y= None):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select top n components
        self.components = eigenvectors[:, :self.n_components]

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance

        return self

    def transform(self, X, y =None):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)