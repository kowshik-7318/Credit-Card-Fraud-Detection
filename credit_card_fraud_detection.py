import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import init, Fore, Style

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

init(autoreset=True)

# Constants
DATA_FILE = 'creditcard.csv'
MODEL_FILE = 'best_model.pkl'
SCALER_FILE = 'scaler.pkl'

class FraudDetection:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        self.model_name = None
        self.results = {}
        
    def print_header(self, title):
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + Style.BRIGHT + f"{title.center(70)}")
        print(Fore.CYAN + "="*70 + "\n")
    
    def load_data(self):
        if not os.path.exists(DATA_FILE):
            print(Fore.RED + f"Error: {DATA_FILE} not found.")
            return False
        
        try:
            self.df = pd.read_csv(DATA_FILE)
            if 'Time' in self.df.columns:
                self.df = self.df.drop('Time', axis=1)
            self.df = self.df.dropna()
            print(Fore.GREEN + f"Loaded {len(self.df)} transactions")
            return True
        except Exception as e:
            print(Fore.RED + f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        if self.df is None:
            print(Fore.RED + "Please load data first.")
            return
        
        self.print_header("DATA EXPLORATION")
        
        total = len(self.df)
        fraud = len(self.df[self.df['Class'] == 1])
        legit = len(self.df[self.df['Class'] == 0])
        
        print(f"Total Transactions     : {total:,}")
        print(f"Legitimate            : {legit:,} ({legit/total*100:.2f}%)")
        print(f"Fraudulent            : {fraud:,} ({fraud/total*100:.4f}%)")
        print(f"Ratio                 : {legit//fraud if fraud > 0 else 0}:1")
        print(f"\nAmount Statistics:")
        print(self.df['Amount'].describe())
        
        # Quick visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        self.df['Class'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
        plt.title('Class Distribution')
        plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
        
        plt.subplot(1, 2, 2)
        self.df[self.df['Class'] == 0]['Amount'].hist(bins=30, alpha=0.7, label='Legit', color='#2ecc71')
        self.df[self.df['Class'] == 1]['Amount'].hist(bins=30, alpha=0.7, label='Fraud', color='#e74c3c')
        plt.title('Amount Distribution')
        plt.xlabel('Amount')
        plt.legend()
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=100)
        print(Fore.GREEN + "\nVisualization saved as 'data_analysis.png'")
        plt.close()
    
    def preprocess(self):
        if self.df is None:
            print(Fore.RED + "Please load data first.")
            return False
        
        try:
            X = self.df.drop('Class', axis=1)
            y = self.df['Class']
            
            self.scaler = StandardScaler()
            X['Amount'] = self.scaler.fit_transform(X[['Amount']])
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Apply SMOTE
            if len(self.y_train.value_counts()) > 1:
                min_samples = min(self.y_train.value_counts())
                k_neighbors = min(5, min_samples - 1)
                if k_neighbors > 0:
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                    print(Fore.GREEN + f"Applied SMOTE. New training size: {len(self.X_train)}")
            
            print(Fore.GREEN + "Preprocessing completed.")
            return True
        except Exception as e:
            print(Fore.RED + f"Preprocessing error: {e}")
            return False
    
    def train_models(self):
        if self.X_train is None:
            print(Fore.RED + "Please preprocess data first.")
            return
        
        self.print_header("MODEL TRAINING")
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
        best_f1 = 0
        best_model = None
        best_name = ""
        
        for name, model in models.items():
            print(Fore.YELLOW + f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc = roc_auc_score(self.y_test, y_pred)
            
            self.results[name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'roc': roc}
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name
        
        self.model = best_model
        self.model_name = best_name
        self.display_results()
        self.save_model()
    
    def display_results(self):
        self.print_header("EVALUATION RESULTS")
        
        print(Style.BRIGHT + f"{'Model':<22} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<8} {'ROC-AUC':<8}")
        print("-" * 75)
        
        for name, metrics in self.results.items():
            print(f"{name:<22} {metrics['acc']:<10.4f} {metrics['prec']:<11.4f} {metrics['rec']:<8.4f} {metrics['f1']:<8.4f} {metrics['roc']:<8.4f}")
        
        print("-" * 75)
        print(Fore.GREEN + Style.BRIGHT + f"\nBest Model: {self.model_name} (F1: {self.results[self.model_name]['f1']:.4f})")
        
        # Confusion matrix visualization
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png', dpi=100)
        plt.close()
        print(Fore.CYAN + "\nConfusion matrix saved as 'confusion_matrix.png'")
    
    def save_model(self):
        try:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump({'model': self.model, 'model_name': self.model_name}, f)
            with open(SCALER_FILE, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(Fore.GREEN + f"Model saved as '{MODEL_FILE}'")
        except Exception as e:
            print(Fore.RED + f"Error saving model: {e}")
    
    def load_model(self):
        try:
            if not os.path.exists(MODEL_FILE):
                print(Fore.RED + "No saved model found. Train first.")
                return False
            
            with open(MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.model_name = data['model_name']
            
            with open(SCALER_FILE, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print(Fore.GREEN + f"Loaded model: {self.model_name}")
            return True
        except Exception as e:
            print(Fore.RED + f"Error loading model: {e}")
            return False
    
    def predict(self):
        if self.model is None:
            print(Fore.RED + "Please train or load a model first.")
            return
        
        self.print_header("FRAUD PREDICTION")
        
        # Input method
        print("Enter 29 values (V1-V28, Amount) or use sample:")
        print("1. Enter manually")
        print("2. Use sample data from dataset")
        
        choice = input("Choice [1/2]: ").strip()
        
        features = []
        amount = 0
        
        if choice == '1':
            try:
                values = input("Enter 29 comma-separated values: ").split(',')
                if len(values) != 29:
                    print(Fore.RED + "Need exactly 29 values")
                    return
                features = [float(x) for x in values[:-1]]
                amount = float(values[-1])
            except:
                print(Fore.RED + "Invalid input. Need numeric values.")
                return
        else:
            # Use sample from dataset
            sample = self.df.sample(1)
            features = sample.drop('Class', axis=1).iloc[0, :-1].tolist()
            amount = sample['Amount'].iloc[0]
            print(Fore.CYAN + f"Using sample transaction: Amount = ${amount:.2f}")
        
        # Make prediction
        input_df = pd.DataFrame([features + [amount]], 
                               columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])
        input_df['Amount'] = self.scaler.transform(input_df[['Amount']])
        
        prob = self.model.predict_proba(input_df)[0][1]
        is_fraud = prob >= 0.5
        
        # Display result
        print("\n" + "="*50)
        if is_fraud:
            print(Back.RED + Fore.WHITE + Style.BRIGHT + " 🚨 FRAUDULENT TRANSACTION ".center(50))
        else:
            print(Back.GREEN + Fore.WHITE + Style.BRIGHT + " ✅ LEGITIMATE TRANSACTION ".center(50))
        print("="*50)
        
        print(f"\nFraud Probability: {prob*100:.2f}%")
        print(f"Legit Probability: {(1-prob)*100:.2f}%")
        print(f"Amount: ${amount:.2f}")
        print(f"Model: {self.model_name}")
        
        # Risk level
        risk_level = "LOW" if prob < 0.25 else "MEDIUM" if prob < 0.50 else "HIGH" if prob < 0.75 else "CRITICAL"
        color = Fore.GREEN if prob < 0.25 else Fore.YELLOW if prob < 0.50 else Fore.RED
        print(f"\nRisk Level: {color}{risk_level}{Style.RESET_ALL}")
        print("="*50)

def main():
    system = FraudDetection()
    
    while True:
        system.print_header("FRAUD DETECTION SYSTEM")
        print("1. Load & Explore Data")
        print("2. Preprocess Data")
        print("3. Train Models")
        print("4. Load Saved Model")
        print("5. Predict Transaction")
        print("6. Exit")
        
        choice = input(Fore.YELLOW + "\nChoice [1-6]: " + Style.RESET_ALL).strip()
        
        if choice == '1':
            if system.load_data():
                system.explore_data()
        elif choice == '2':
            system.preprocess()
        elif choice == '3':
            system.train_models()
        elif choice == '4':
            system.load_model()
        elif choice == '5':
            system.predict()
        elif choice == '6':
            print(Fore.GREEN + "Goodbye!")
            sys.exit(0)
        else:
            print(Fore.RED + "Invalid choice.")
        
        input("\nPress Enter to continue...")
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists(DATA_FILE):
        print(Fore.RED + f"Warning: {DATA_FILE} not found.")
        print("Please place the creditcard.csv dataset in the current directory.")
        print("You can download it from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        input("\nPress Enter to continue...")
    
    main()