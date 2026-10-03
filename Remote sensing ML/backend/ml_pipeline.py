import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional

class MineralProspectivityPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.features = []

    def train(self, df_data: List[Dict], features: List[str], model_type: str = 'random_forest', hyperparams: Optional[Dict] = None) -> Dict[str, Any]:
        """Trains a machine learning model for mineral prospectivity mapping."""
        df = pd.DataFrame(df_data)
        self.features = features
        
        # Prepare inputs and labels
        X = df[features].copy()
        y = df['deposit'].copy()
        
        # Split into training and testing sets
        # Stratify because y is highly imbalanced
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Simple random oversampling of the positive class to handle severe imbalance
        pos_indices = np.where(y_train == 1)[0]
        neg_indices = np.where(y_train == 0)[0]
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            # Oversample positive class to match about 25% of negative class size
            target_pos_count = len(neg_indices) // 3
            multiplier = target_pos_count // len(pos_indices)
            if multiplier > 1:
                oversampled_pos_idx = np.random.choice(pos_indices, size=target_pos_count, replace=True)
                X_train_pos = X_train.iloc[oversampled_pos_idx]
                y_train_pos = y_train.iloc[oversampled_pos_idx]
                
                X_train = pd.concat([X_train.iloc[neg_indices], X_train_pos], axis=0)
                y_train = pd.concat([y_train.iloc[neg_indices], y_train_pos], axis=0)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select and initialize model
        params = hyperparams or {}
        if model_type == 'random_forest':
            n_estimators = int(params.get('n_estimators', 100))
            max_depth = int(params.get('max_depth', 8)) if params.get('max_depth') else None
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            n_estimators = int(params.get('n_estimators', 100))
            learning_rate = float(params.get('learning_rate', 0.1))
            max_depth = int(params.get('max_depth', 4))
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators, 
                learning_rate=learning_rate, 
                max_depth=max_depth,
                random_state=42
            )
        elif model_type == 'svm':
            C = float(params.get('C', 1.0))
            gamma = params.get('gamma', 'scale')
            self.model = SVC(
                C=C, 
                gamma=gamma, 
                probability=True, 
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Get probability outputs
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            # Fallback for models without direct predict_proba (though SVC probability=True has it)
            y_prob = y_pred.astype(float)
            
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.5
            
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            'tn': int(cm[0, 0]) if cm.shape == (2, 2) else int(cm[0, 0]),
            'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0
        }
        
        # ROC Curve coordinates
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            # Downsample ROC curve details for faster API response
            step = max(1, len(fpr) // 30)
            roc_data = [{'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr[::step], tpr[::step])]
            # Make sure we include the final point (1,1)
            if roc_data[-1] != {'fpr': 1.0, 'tpr': 1.0}:
                roc_data.append({'fpr': 1.0, 'tpr': 1.0})
        except Exception:
            roc_data = [{'fpr': 0.0, 'tpr': 0.0}, {'fpr': 1.0, 'tpr': 1.0}]
            
        # Feature importances (available for Tree-based models)
        importances = {}
        # Safely access tree-based feature importances. Some models (e.g. SVC) do not
        # have `feature_importances_`, so use getattr to avoid static analysis issues
        # and ensure correct runtime behavior.
        fi = getattr(self.model, 'feature_importances_', None)
        if fi is not None:
            for feat, val in zip(self.features, fi):
                importances[feat] = float(val)
        else:
            # For SVM, use permutation importance or a simple correlation baseline
            # We will use absolute correlation coefficients as dummy importance weights
            for feat in self.features:
                importances[feat] = float(np.abs(np.corrcoef(df[feat], df['deposit'])[0, 1]))
            # Normalise to sum to 1
            total = sum(importances.values()) or 1.0
            for feat in importances:
                importances[feat] /= total
                
        # Sort importances
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        importances_list = [{'feature': k, 'importance': v} for k, v in sorted_importances]
        
        return {
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc)
            },
            'confusion_matrix': cm_dict,
            'roc_curve': roc_data,
            'feature_importances': importances_list
        }

    def predict_grid(self, df_data: List[Dict]) -> List[float]:
        """Computes mineral prospectivity probabilities for the entire spatial grid."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        df = pd.DataFrame(df_data)
        X = df[self.features]
        X_scaled = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return [float(p) for p in probs]
