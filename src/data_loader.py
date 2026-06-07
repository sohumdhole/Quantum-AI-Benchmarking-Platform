import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

class DigitalTransformationDataset:
    """
    Generates and preprocesses digital transformation datasets suitable for 
    classical and quantum machine learning models.
    """
    
    def __init__(self, dataset_type="predictive_maintenance", n_samples=200, n_features=6, noise=0.1, random_state=42):
        self.dataset_type = dataset_type
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.random_state = random_state
        
        self.raw_df = None
        self.X = None
        self.y = None
        
        self._generate_dataset()
        
    def _generate_dataset(self):
        """Generates synthetic tabular data matching digital transformation scenarios."""
        np.random.seed(self.random_state)
        
        if self.dataset_type == "predictive_maintenance":
            # Predictive Maintenance: Predict machine failure (0 = Normal, 1 = Fail)
            # Features: Temp, Vibration, Rotational Speed, Pressure, Run Hours, Age
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=max(2, self.n_features - 2),
                n_redundant=min(2, self.n_features // 3),
                n_clusters_per_class=1,
                weights=[0.7, 0.3],  # Imbalanced class typical for failures
                flip_y=self.noise,
                random_state=self.random_state
            )
            feature_names = [f"Sensor_{i}_Vibration" if i%2==0 else f"Sensor_{i}_Temp" for i in range(self.n_features)]
            if self.n_features >= 6:
                feature_names[:6] = ["Temperature_C", "Vibration_mm_s", "Rotational_Speed_RPM", "Pressure_kPa", "Operating_Hours", "Equipment_Age_Years"]
                
        elif self.dataset_type == "financial_fraud":
            # Financial Fraud Detection (0 = Legitimate, 1 = Fraud)
            # Features: Tx Amount, Dist from Home, Device Score, Transaction Speed, Hour of Day, Tx Count
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=max(2, self.n_features - 1),
                n_redundant=0,
                n_clusters_per_class=1,
                weights=[0.85, 0.15],
                flip_y=self.noise,
                random_state=self.random_state
            )
            feature_names = [f"Tx_Metric_{i}" for i in range(self.n_features)]
            if self.n_features >= 6:
                feature_names[:6] = ["Transaction_Amount_USD", "Distance_From_Billing_Address", "Device_Risk_Score", "Velocity_1h", "Hour_of_Day", "Historical_Fraud_Rate"]
                
        elif self.dataset_type == "customer_churn":
            # Telecom/SaaS Customer Churn (0 = Retained, 1 = Churned)
            # Features: Monthly Bill, Contract Duration, Total Usage, Support Calls, Account Age, Tenure
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=max(2, self.n_features - 1),
                n_redundant=0,
                n_clusters_per_class=1,
                weights=[0.65, 0.35],
                flip_y=self.noise,
                random_state=self.random_state
            )
            feature_names = [f"Usage_Metric_{i}" for i in range(self.n_features)]
            if self.n_features >= 6:
                feature_names[:6] = ["Monthly_Charges_USD", "Contract_Type_Months", "Usage_GB", "Customer_Service_Calls", "Account_Age_Months", "Loyalty_Score"]
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
        self.X = X
        self.y = y
        
        # Build DataFrame
        self.raw_df = pd.DataFrame(X, columns=feature_names[:self.n_features])
        self.raw_df["Target"] = y

    def get_data(self):
        """Returns the raw dataframe."""
        return self.raw_df

    def preprocess(self, test_size=0.25, pca_components=None, scale_quantum=True):
        """
        Splits, reduces dimensionality, and scales data.
        If scale_quantum is True, scales to [0, pi] for angle embedding.
        Otherwise scales to [0, 1] or standardizes.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )
        
        # Apply PCA if requested to reduce qubits requirement
        if pca_components is not None and pca_components < self.n_features:
            pca = PCA(n_components=pca_components, random_state=self.random_state)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            
        # Scaling
        if scale_quantum:
            # Map features to [0, pi] for quantum angle embeddings
            scaler = MinMaxScaler(feature_range=(0, np.pi))
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
