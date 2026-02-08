# MLP training from scratch
import numpy as np

def train_model(model, X_train, y_train, loss_fn, optimizer, epochs=50, batch_size=64, patience=5, min_delta=1e-4):

    # Get the number of samples in the training data
    num_samples = X_train.shape[0]
    loss_history = [] # To store loss at each epoch

    best_loss = float('inf') # Initialize best loss to infinity for early stopping
    patience_counter = 0 # Counter to track epochs without improvement for early stopping

    
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples) # Shuffle the training data
        X_train = X_train[indices] # Shuffle the inputs
        y_train = y_train[indices] # Shuffle the labels

        epoch_loss = 0.0 # Initialise loss for the epoch
        num_batches = 0 # To count number of batches processed

        # Mini-batch training
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples) # Ensure we don't go out of bounds
            X_batch = X_train[start_idx:end_idx] # Get the current mini-batch
            y_batch = y_train[start_idx:end_idx] # Get the corresponding labels

            y_pred = model.forward(X_batch) # Forward pass

            loss = loss_fn.forward(y_pred, y_batch) # Compute loss
            epoch_loss += loss # Accumulate loss over batches
            num_batches += 1 # Increment batch count

            dloss = loss_fn.backward() # Backward pass to get gradient of loss w.r.t. predictions
            model.backward(dloss) # Backpropagate through the model

            params, grads = model.get_parameters_and_grads() # Get model parameters and their local gradients
            optimizer.step(params, grads) # Update model parameters using local gradients

        avg_epoch_loss = epoch_loss / num_batches # Average loss for the epoch
        loss_history.append(avg_epoch_loss) # Store average loss

        # Early stopping check
        # If the improvement in loss is greater than min_delta, reset the patience counter and update best_loss with the new best loss
        if best_loss - avg_epoch_loss > min_delta:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1 # Increment patience counter if no significant improvement

        # Print loss every 10 epochs and at the first epoch for tracking training progress
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

        # If the patience counter exceeds the specified patience, stop training early to prevent overfitting
        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch+1} "
                f"(no improvement > {min_delta} for {patience} epochs)"
            )
            break

    return loss_history

