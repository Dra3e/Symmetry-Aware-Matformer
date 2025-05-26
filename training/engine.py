from pathlib import Path
import re
from typing import Callable, List
import torch
import datetime
from models.Transformer import Transformer
from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from training.loss_functions import AutomaticWeightedLoss, LossProperty, LossContrastive_Vectorized
from training.data_getter import DataGetter
from tqdm import tqdm

class SAM_TrainingEngine:
    """
    Training engine for the SAM model, which handles the training and evaluation process.
    """
    def __init__(self, learning_rate=3e-4, weight_decay=1e-2):
        """
        Initialize the training engine.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Transformer().to(self.device)
        self.multi_loss = AutomaticWeightedLoss(n_tasks=2).to(self.device)
        self.model_optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(),         "weight_decay": weight_decay},
                {"params": self.multi_loss.parameters(),    "weight_decay": 0.0},  # do NOT decay σ
            ],
            lr=learning_rate,
            betas=(0.9, 0.98),
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
    
    def fit(self, property, training_path: str, validation_path: str, graph_builder: Callable[[Crystal], list], loss_type = "multi", evaluation_loss_type="mse", num_epochs = 50, batch_size = 32, margin = 5.0):
        """
        Train the model on the given dataset.

        Args:
            property (str): Target property to predict.
            training_path (str): Path to the training dataset.
            validation_path (str): Path to the validation dataset.
            graph_builder (Callable): Function to build graphs from crystals.
            loss_type (str): Type of loss to use ("multi", "mse", or "contrastive").
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            margin (float): margin for contrastive loss.
        """
        # Create training and validation datasets
        training_data_getter = DataGetter(
                path           = training_path,
                property_name  = property,
                graph_builder  = graph_builder,
                batch_size     = batch_size,
                shuffle        = True,         # shuffle only on the training split
        )
        training_dataloader = training_data_getter.loader
        eval_dataloader = DataGetter(
                path           = validation_path,
                property_name  = property,
                graph_builder  = graph_builder,
                batch_size     = batch_size,
                shuffle        = False,        # keep order for evaluation
        ).loader
        best_eval_loss = float('inf')           # Track the best validation loss
        best_model_state = None                 # Store the best model state

        # Training loop
        for epoch in range(num_epochs):
            self.model.train() # Set model to training mode
            epoch_loss = 0.0 # Initialize epoch loss

            for graphs, target_values in tqdm(training_dataloader, desc="Training on crystals", unit="batch"):
                # Move crystals and targets to device
                graphs, target_values = self._move_to_device(graphs, target_values)
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda'), dtype=torch.bfloat16): # Mixed precision
                    # Forward pass and compute loss
                    property_loss, contrastive_loss = self._get_loss(
                        graph_batch=graphs,
                        loss_type=loss_type,
                        target_values=target_values,
                        margin=margin
                    )
                if loss_type == "multi":
                    self._pcgrad(property_loss, contrastive_loss) # PCGrad on automatic Weighted Loss
                    batch_loss = (property_loss + contrastive_loss).detach().item()
                else:

                    solo = property_loss + contrastive_loss # Because in htis case one of the losses is zero
                    self.model_optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(solo).backward()
                    self.scaler.step(self.model_optimizer)
                    self.scaler.update()
                    batch_loss = solo.detach().item()
                epoch_loss += batch_loss


            # Evaluate validation loss
            eval_loss = self.run(
                dataloader=eval_dataloader,
                loss_type=evaluation_loss_type,
                margin=margin
            )
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss / len(training_dataloader):.4f}, Eval Loss: {eval_loss:.4f}")

            # Save the best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state = self.model.state_dict().copy()

        # Save the best model
        if best_model_state:
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            name = f"training/results/{property}_{date}.pth"
            torch.save({
                'margin': margin,
                'property': property,
                'model_state_dict': best_model_state,
                "multi_loss_state_dict": self.multi_loss.state_dict()
            }, name)
            print(f"Best model saved with Val Loss: {best_eval_loss:.4f}")

        print("Training Complete!")
        return best_eval_loss

    def run(self, dataloader, loss_type = "mse", margin=5.0):
        """
        Evaluate the model on a dataset.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            loss_type (str): Type of loss to use ("mse" or "contrastive").
            margin (float): margin for contrastive loss.

        Returns:
            float: Average loss over the dataset.
        """
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        sample_count = 0

        with torch.no_grad():
            for graphs, target_values in dataloader:
                # Move graphs and targets to device
                graphs = [[g.to(self.device) for g in g_list] for g_list in graphs]  # Move each graph to the device
                target_values = target_values.to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda'), dtype=torch.bfloat16):
                    # Forward pass and compute loss
                    property_loss, contrastive_loss = self._get_loss(
                        graph_batch=graphs, 
                        loss_type=loss_type, 
                        target_values=target_values, 
                        margin=margin
                    )
                    if loss_type=="multi":
                        total = self.multi_loss(property_loss, contrastive_loss)
                    else: # Because otherwise one of the losses is zero
                        total = property_loss + contrastive_loss
                    
                    total_loss += total.item()
                    sample_count += 1
                    

        return total_loss / sample_count if sample_count > 0 else float('inf')
    
    def _get_loss(self, loss_type, graph_batch, target_values, margin=5.0):
        """
        Compute the loss for a batch of graphs.

        Args:
            loss_type (str): Type of loss to use ("multi", "mse", or "contrastive").
            graph_batch (list): Batch of graphs.
            target_values (torch.Tensor): Target values for the batch.
            margin (float): margin for contrastive loss.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        results = [self.model(graphs=graph_list) for graph_list in graph_batch]
        batch_property_losses = []
        batch_contrastive_losses = []
        valid_grah_index = -1 # The last graph in the batch is the valid graph (e.g. the crystal)

        for i, (predictions, latents, graph_list) in enumerate(results):
            target = target_values[i]

            valid_prediction = predictions[valid_grah_index] #Get the prediction of the valid graph e.g. the crystal
            loss_property = LossProperty(
                actual_Y = target.expand_as(valid_prediction),
                predicted_Y = valid_prediction
            )

            
            flags = torch.tensor([g.m_valid for g in graph_list], dtype=torch.float32, device=self.device)
            lat_slice   = latents[:valid_grah_index]
            flag_slice  = flags[:valid_grah_index]
            if lat_slice.numel() == 0:
                mean_loss_contrastive = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                mean_loss_contrastive = LossContrastive_Vectorized(
                    latent_ref = latents[valid_grah_index],
                    latents    = lat_slice,
                    flags      = flag_slice,
                    margin     = margin,
            )

            batch_property_losses.append(loss_property)
            batch_contrastive_losses.append(mean_loss_contrastive)

        # Aggregate losses
        loss_property_batch = torch.stack(batch_property_losses).mean()
        loss_contrastive_batch = torch.stack(batch_contrastive_losses).mean()
        
        # Handle different loss types
        zero = torch.tensor(0.0, device=self.device)
        if loss_type == "multi":
            # return *both* raw task losses; AWL will be applied later
            return loss_property_batch, loss_contrastive_batch
        elif loss_type == "mse":
            return loss_property_batch, zero
        elif loss_type == "contrastive":
            return zero, loss_contrastive_batch
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _pcgrad(self, property_loss, contrastive_loss):
        """
        PCGrad + AutomaticWeightedLoss (AWL) with AMP
        Fixed: only one unscale_() call, no glued gradients.
        """
        # ------------------------------------------------------------------ 1) AWL weights
        sigma_sq = torch.exp(self.multi_loss.log_sigma).pow(2)        # σ_i² ≥ 0
        w_prop, w_con = (0.5 / sigma_sq).unbind()                     # two scalars

        # ------------------------------------------------------------------ 2) set-up
        self.model_optimizer.zero_grad(set_to_none=True)
        scale = self.scaler.get_scale()                               # same for both tasks

        # ------------------------------------------------------------------ 3) property-loss backward (scaled)
        scaled_prop = self.scaler.scale(w_prop * property_loss)
        scaled_prop.backward(retain_graph=True)

        # cloned *scaled* gradients
        g_prop_scaled = [self._safe_clone(p.grad, p) for p in self.model.parameters()]

        # ------------------------------------------------------------------ 4) contrastive-loss backward (scaled)
        for p in self.model.parameters():                             # clear before 2nd pass
            p.grad = None
        scaled_con = self.scaler.scale(w_con * contrastive_loss)
        scaled_con.backward(retain_graph=True)

        # ------------------------------------------------------------------ 5) unscale ONCE (affects current .grad tensors)
        self.scaler.unscale_(self.model_optimizer)

        # un-scaled contrastive grads (already in .grad after unscale_)
        g_con = [self._safe_clone(p.grad, p) for p in self.model.parameters()]

        # un-scaled property grads = scaled / scale
        inv_scale = 1.0 / scale
        g_prop = [gp * inv_scale for gp in g_prop_scaled]

        # ------------------------------------------------------------------ 6) PCGrad projection & merge  (in-place on un-scaled grads)
        for p, gp, gc in zip(self.model.parameters(), g_prop, g_con):
            dot = (gp * gc).sum()
            if dot < 0:                                               # conflicting
                gc = gc - dot / gp.norm().pow(2) * gp
            p.grad = gp + gc                                          # merged gradient

        # ------------------------------------------------------------------ 7) AWL regulariser   log(1+σ²)
        # Only σ-parameters need grads here, so we add it *without* scaling.
        torch.log1p(sigma_sq).sum().backward()

        # ------------------------------------------------------------------ 8) optimiser / scaler step
        self.scaler.step(self.model_optimizer)
        self.scaler.update()
    
    def _safe_clone(self, gradient, parameter: torch.nn.Parameter):
        """Return a detached clone of `g`, or a zero tensor if g is None."""
        if gradient is None:
            return torch.zeros_like(parameter, memory_format=torch.preserve_format)
        return gradient.detach().clone()

    def _move_to_device(self, graphs: List[List[CrystalGraph]], targets: torch.Tensor):
        """
        Move graphs and targets to the configured device.

        Args:
            graphs (list): List of graphs.
            targets (torch.Tensor): Target values.

        Returns:
            tuple: Graphs and targets moved to the device.
        """        
        moved_graphs = [[g.to(self.device) for g in g_list] for g_list in graphs]
        return moved_graphs, targets.to(self.device)
        