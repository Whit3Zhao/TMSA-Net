import numpy as np
import torch
import cv2
import config


# Perform data augmentation for the given data and labels
def data_augmentation(data, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aug_data = []
    aug_label = []

    N, C, T = data.shape  # Extract dimensions (batch size, channels, time steps)
    seg_size = T // config.num_segs  # Calculate segment size
    aug_data_size = config.batch_size // 4  # Determine augmentation data size

    for cls in range(4):  # Iterate through each class
        cls_idx = torch.where(label == cls)
        cls_data = data[cls_idx]
        data_size = cls_data.shape[0]
        if data_size == 0 or data_size == 1:  # Skip if there is no or insufficient data
            continue
        temp_aug_data = torch.zeros((aug_data_size, C, T), device=device)
        for i in range(aug_data_size):  # Generate augmented data
            rand_idx = torch.randint(0, data_size, (config.num_segs,), device=device)
            for j in range(config.num_segs):
                temp_aug_data[i, :, j * seg_size:(j + 1) * seg_size] = cls_data[rand_idx[j], :,
                                                                               j * seg_size:(j + 1) * seg_size]
        aug_data.append(temp_aug_data)
        aug_label.extend([cls] * aug_data_size)

    if len(aug_data) == 0:  # If no augmentation data, return original data
        return data, label

    aug_data = torch.cat(aug_data, dim=0)
    aug_label = torch.tensor(aug_label, device=device)
    aug_shuffle = torch.randperm(len(aug_data), device=device)
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    return aug_data, aug_label


# Class to extract activations and register gradients for targeted layers
class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []  # List to store gradients
        self.activations = []  # List to store activations
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            # Register forward and backward hooks
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    # Save activations during the forward pass
    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    # Save gradients during the backward pass
    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    # Forward pass through the model
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    # Release all hooks
    def release(self):
        for handle in self.handles:
            handle.remove()


# Class for GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layers, reshape_transform=None, use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    # Calculate weights for CAM using gradients
    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=2, keepdims=True)

    # Compute the loss based on target categories
    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss += output[i, target_category[i]]
        return loss

    # Generate CAM image by aggregating weighted activations
    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    # Get the width and height of the input tensor
    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    # Compute CAM for each target layer
    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # Remove negative values
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    # Aggregate CAMs from multiple layers
    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    # Scale CAM to a specific size
    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)
        return result

    # Generate CAMs by forward and backward passes
    def __call__(self, input_tensor, target_category):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)  # Forward pass
        target_list = [target_category] * input_tensor.size(0)

        self.model.zero_grad()
        loss = self.get_loss(output, target_list)  # Compute loss
        print('The loss is', loss)
        loss.backward(retain_graph=True)  # Backward pass

        cam_per_layer = self.compute_cam_per_layer(input_tensor)  # Compute CAMs per layer
        return self.aggregate_multi_layers(cam_per_layer)  # Aggregate CAMs

    # Release resources on deletion
    def __del__(self):
        self.activations_and_grads.release()

    # Context manager methods for resource management
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(f"An exception occurred in CAM block: {exc_type}. Message: {exc_value}")
            return True
