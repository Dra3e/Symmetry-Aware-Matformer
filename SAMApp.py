import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.trainer import run, graph_collate
from training.evaluator import compute_loss
from models.Transformer import Transformer
from training.dataset import get_training_data_from_pkl, CrystalGraphDataset

# Paths
DATA_PATH = "data"
WEIGHTS_PATH = "training/results"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging CUDA errors

class SAMApp:
    """Main application class for SAM GUI."""
    def __init__(self, root):
        self.root = root                                                            # Main window
        self.root.title("SAM (Symmetry Aware Matformer)")                           # Window title
        #self.root.iconbitmap("icon.ico")                                            # Window icon
        self.root.geometry("800x600")                                               # Window size
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for PyTorch

        
        # Check/create folders
        self.setup_folders()
        
        # Property mapping
        self.property_mapping = {
            "formation energy": "e_hull",                                         
            "band gap": "gap pbe",
            "magnetic moment": "mu_b",
            "elastic anisotropy": "elastic anisotropy",
            "bulk modulus": "bulk modulus",
            "shear modulus": "shear modulus"
        }
        
        # GUI Setup
        self.setup_gui()                                                            
    
    def setup_folders(self):
        """Ensure data and weights folders exist."""
        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(WEIGHTS_PATH, exist_ok=True)
    
    def setup_gui(self):
        """Main GUI layout."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="SAM - Crystal Property Predictor",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Select Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            mode_frame,
            text="Train New Model",
            command=self.open_training_window
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            mode_frame,
            text="Run Existing Model",
            command=self.open_run_window
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            mode_frame,
            text="Exit",
            command=self.root.quit
        ).pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(fill=tk.X, pady=(10, 0))
    
    def open_training_window(self):
        """Open training configuration window."""
        train_window = tk.Toplevel(self.root)
        train_window.title("Train New Model")
        
        # Data selection
        data_files = self.get_available_files(DATA_PATH)
        if not data_files:
            messagebox.showerror("Error", "No data files found in 'data' folder!")
            train_window.destroy()
            return
        
        ttk.Label(train_window, text="Training Dataset:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.train_data_combo = ttk.Combobox(train_window, values=data_files)
        self.train_data_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(train_window, text="Validation Dataset:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.val_data_combo = ttk.Combobox(train_window, values=data_files)
        self.val_data_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Property selection
        ttk.Label(train_window, text="Property to Predict:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.property_combo = ttk.Combobox(train_window, values=list(self.property_mapping.keys()))
        self.property_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Training parameters
        ttk.Label(train_window, text="Epochs:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.epochs_entry = ttk.Entry(train_window)
        self.epochs_entry.insert(0, "50")
        self.epochs_entry.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(train_window, text="Batch Size:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.batch_entry = ttk.Entry(train_window)
        self.batch_entry.insert(0, "32")
        self.batch_entry.grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(train_window, text="Learning Rate:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.lr_entry = ttk.Entry(train_window)
        self.lr_entry.insert(0, "0.01")
        self.lr_entry.grid(row=5, column=1, padx=5, pady=5)
        
        # Start button
        ttk.Button(
            train_window,
            text="Start Training",
            command=self.start_training
        ).grid(row=6, column=0, columnspan=2, pady=10)
    
    def open_run_window(self):
        """Open model running window."""
        run_window = tk.Toplevel(self.root)
        run_window.title("Run Existing Model")
        
        # Model selection
        model_files = self.get_available_files(WEIGHTS_PATH)
        if not model_files:
            messagebox.showerror("Error", "No model files found in 'training/results'!")
            run_window.destroy()
            return
        
        ttk.Label(run_window, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_combo = ttk.Combobox(run_window, values=model_files)
        self.model_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Data selection
        data_files = self.get_available_files(DATA_PATH)
        ttk.Label(run_window, text="Validation Dataset:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.run_data_combo = ttk.Combobox(run_window, values=data_files)
        self.run_data_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Run button
        ttk.Button(
            run_window,
            text="Evaluate Model",
            command=self.run_model
        ).grid(row=2, column=0, columnspan=2, pady=10)
    
    def get_available_files(self, directory):
        """Get files from a directory."""
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    def start_training(self):
        """Start training with selected parameters."""
        try:
            training_file = self.train_data_combo.get()
            val_file = self.val_data_combo.get()
            property_name = self.property_combo.get()
            
            if not all([training_file, val_file, property_name]):
                messagebox.showerror("Error", "Please fill all fields!")
                return
            
            self.status_var.set("Training started...")
            
            for epoch in tqdm(range(int(self.epochs_entry.get())), desc="Training Progress", unit="epoch"):
                run(
                    self.property_mapping[property_name],
                    training_path=os.path.join(DATA_PATH, training_file),
                    validation_path=os.path.join(DATA_PATH, val_file),
                    num_epochs=int(self.epochs_entry.get()),
                    batch_size=int(self.batch_entry.get()),
                    learning_rate=float(self.lr_entry.get()),
                    threshold=5.0  # You can add this as an input too
                )
            
            self.status_var.set("Training completed!")
            messagebox.showinfo("Success", "Training finished successfully!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_var.set("Error in training")
    
    def run_model(self):
        """Run evaluation on selected model."""
        try:
            model_file = self.model_combo.get()
            data_file = self.run_data_combo.get()
            
            if not all([model_file, data_file]):
                messagebox.showerror("Error", "Please select both model and dataset!")
                return
            
            self.status_var.set("Loading model...")
            
            # Load model
            checkpoint = torch.load(
                os.path.join(WEIGHTS_PATH, model_file),
                map_location=self.device
            )
            model = Transformer().to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            factors = checkpoint['factors'].to(self.device)
            target_property = checkpoint['property']
            threshold = checkpoint['threshold']
            
            # Load data
            dataset, target_properties = get_training_data_from_pkl(
                target_property,
                os.path.join(DATA_PATH, data_file)
            )
            graphs = CrystalGraphDataset(dataset, target_properties)
            dataloader = DataLoader(
                graphs,
                batch_size=25,
                shuffle=False,
                collate_fn=graph_collate,
                num_workers=4,
                pin_memory=True
            )
            
            # Compute loss
            self.status_var.set("Evaluating model...")
            with tqdm(total=len(dataloader), desc="Evaluation Progress", unit="batch") as pbar:
                eval_loss = compute_loss(model, dataloader, factors, self.device, threshold)
            
            messagebox.showinfo(
                "Results",
                f"Evaluation completed!\nMSA: {eval_loss}"
            )
            self.status_var.set(f"Evaluation done. MSA: {eval_loss}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")
            self.status_var.set("Error in evaluation")

if __name__ == "__main__":
    root = tk.Tk()
    app = SAMApp(root)
    root.mainloop()