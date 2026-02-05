# MLP training from scratch
import numpy as np

def train_model(model, X_train, y_train, loss_fn, optimizer,
          epochs=50, batch_size=64):

    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            # Forward
            y_pred = model.forward(X_batch)

            # Loss
            loss = loss_fn.forward(y_pred, y_batch)
            epoch_loss += loss

            # Backward
            dLoss = loss_fn.backward()
            model.backward(dLoss)

            # Update
            params, grads = model.get_parameters_and_grads()
            optimizer.step(params, grads)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
