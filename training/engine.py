from typing import Callable, List
import torch
import datetime
from models.Transformer import Transformer
from models.components.crystal import Crystal
from models.components.crystal_graph import CrystalGraph
from training.loss_functions import MultiTaskLoss
from training.data_getter import DataGetter
from tqdm import tqdm

class SAM_TrainingEngine:
    def __init__(self, learning_rate: float = 1e-2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Transformer().to(self.device)
        self.multi_loss = MultiTaskLoss(n_tasks=2).to(self.device)
        self.model_optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.multi_loss.parameters()), 
            lr=learning_rate
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
    
    def fit(self, property, training_path: str, validation_path: str, graph_builder: Callable[[Crystal], list], loss_type = "multi", num_epochs = 50, batch_size = 32, threshold = 5.0):
        # Create dataset
        training_data_getter = DataGetter(
                path           = training_path,
                property_name  = property,
                graph_builder  = graph_builder,
                batch_size     = batch_size,
                shuffle        = True,         # shuffle only on the training split
        )
        training_dataloader = training_data_getter.loader
        num_data_points = training_data_getter.num_data_points
        eval_dataloader = DataGetter(
                path           = validation_path,
                property_name  = property,
                graph_builder  = graph_builder,
                batch_size     = batch_size,
                shuffle        = False,        # keep order for evaluation
        ).loader
        best_eval_loss = float('inf')
        best_model_state = None

        # Training loop
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(num_epochs):
                self.model.train() # Set model to training mode
                epoch_loss = 0.0 # Initialize epoch loss

                for graphs, target_values in tqdm(training_dataloader, desc="Training on crystals", unit="batch"):
                    print() # newline
                    # Move crystals and targets to device
                    graphs, target_values = self._move_to_device(graphs, target_values)
                    with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                        # Forward pass
                        model_loss = self._get_loss(
                            graph_batch=graphs, 
                            loss_type=loss_type, 
                            target_values=target_values, 
                            threshold=threshold
                        )
                    # Backprop and optimize
                    self.model_optimizer.zero_grad()
                    if model_loss.requires_grad and model_loss.grad_fn is not None:
                        self.scaler.scale(model_loss).backward()
                        self.scaler.step(self.model_optimizer)
                        self.scaler.update()
                    else:
                        print("⚠️ Skipping batch: No gradient flow detected.")

                    epoch_loss += model_loss.item()

            
                # Evaluate validation loss
                eval_loss = self.run(self.model, eval_dataloader, self.multi_loss, self.device, threshold)
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss / num_data_points:.4f}, Eval Loss: {eval_loss:.4f}")

                # Save the best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model_state = self.model.state_dict().copy()

        # Save the best model
        if best_model_state:
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
            name = f"training/results/{property}_{date}.pth"
            torch.save({
                'threshold': threshold,
                'property': property,
                'model_state_dict': best_model_state,
                "multi_loss_state_dict": self.multi_loss.state_dict()
            }, name)
            print(f"Best model saved with Val Loss: {best_eval_loss:.4f}")

        print("Training Complete!")

    def run(self, dataloader, loss_type = "mse", threshold=5.0):
        self.model.eval()
        total_loss = 0.0
        sample_count = 0

        with torch.no_grad():
            for graphs, target_values in dataloader:
                # Move graphs and targets to device
                graphs = [[g.to(self.device) for g in g_list] for g_list in graphs]  # Move each graph to the device
                target_values = target_values.to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    # Forward pass
                    total = self._get_loss(
                        graph_batch=graphs, 
                        loss_type=loss_type, 
                        target_values=target_values, 
                        threshold=threshold
                    )
                    total_loss += total.item()
                    sample_count += 1

        return total_loss / sample_count if sample_count > 0 else float('inf')
    
    def _get_loss(self, loss_type, graph_batch, target_values, threshold=5.0):
        results = [self.model(graph_list) for graph_list in graph_batch]
        batch_property_losses = []
        batch_contrastive_losses = []

        for i, (predictions, latents, graph_list) in enumerate(results):
            target = target_values[i]
            latent_ref = latents[-1]

            flags = torch.tensor([g.m_valid for g in graph_list], dtype=torch.float32, device=self.device)

            loss_property = MultiTaskLoss.LossProperty(actual_Y=target.expand_as(predictions[-1]), predicted_Y=predictions[-1])
            mean_loss_contrastive = MultiTaskLoss.LossContrastive_Vectorized(
                latent_ref=latent_ref,
                latents=latents,
                flags=flags,
                threshold=threshold
            )

            batch_property_losses.append(loss_property)
            batch_contrastive_losses.append(mean_loss_contrastive)

        loss_property_batch = torch.stack(batch_property_losses).mean()
        loss_contrastive_batch = torch.stack(batch_contrastive_losses).mean()

        if loss_type == "multi":
            print("Property loss:", loss_property_batch.item())
            print("Contrastive loss:", loss_contrastive_batch.item())
            print("Log vars:", self.multi_loss.log_vars.detach().cpu().numpy())
            print("Multi factor loss:", self.multi_loss(loss_property_batch, loss_contrastive_batch).item())

            return self.multi_loss(loss_property_batch, loss_contrastive_batch)
        elif loss_type == "mse":
            return loss_property_batch
        elif loss_type == "contrastive":
            return loss_contrastive_batch
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _move_to_device(self, graphs: List[List[CrystalGraph]], targets: torch.Tensor):
        """Recursively move graphs + targets to the configured *device*."""
        moved_graphs = [[g.to(self.device) for g in g_list] for g_list in graphs]
        return moved_graphs, targets.to(self.device)
        