from data import data
import models
import utils

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xgboost import XGBClassifier

class GlobalLossAttack:

    def __init__(self, dataset_name, model_type, batch_size, device, n_loss_samples=1):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device
        self.n_loss_samples = n_loss_samples

    def load_model(self, index):
        model, _ = utils.load_model(
            dataset=self.dataset_name,
            model_type=self.model_type,
            index_model=index,
            device=self.device,
            n_loss_samples=self.n_loss_samples,
        )
        return model

    @torch.inference_mode()
    def loss_signal(self, audit_loader, index_model):
        model = self.load_model(index_model)
        sig = []
        for samples in audit_loader:
            samples = samples.to(self.device)
            loss = model.per_sample_loss(samples).cpu()
            sig.append(loss)
        sig = torch.concat(sig, dim=0)
        assert sig.shape == (len(audit_loader.dataset),)
        return sig

    def run_attack(self, audit_samples, index_target_model):
        audit_loader = DataLoader(audit_samples, batch_size=self.batch_size, shuffle=False)
        return -self.loss_signal(audit_loader, index_target_model)

class BASE:

    def __init__(self, dataset_name, model_type, batch_size, device, index_ref_models, partition_fns, prior=0.5, n_loss_samples=1):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device
        self.index_ref_models = index_ref_models
        self.partition_fns = partition_fns
        self.prior = prior
        self.n_loss_samples=n_loss_samples

    def load_model(self, index):
        model, _ = utils.load_model(
            dataset=self.dataset_name,
            model_type=self.model_type,
            index_model=index,
            device=self.device,
            n_loss_samples=self.n_loss_samples,
        )
        return model

    @torch.inference_mode()
    def loss_signal(self, audit_loader, index_model):
        model = self.load_model(index_model)
        sig = []
        for samples in audit_loader:
            samples = samples.to(self.device)
            loss = model.per_sample_loss(samples).cpu()
            log_Z = self.partition_fns[index_model]
            sig.append(loss + log_Z)
        sig = torch.concat(sig, dim=0)
        assert sig.shape == (len(audit_loader.dataset),)
        return sig

    def run_attack(self, audit_samples, index_target_model):
        assert index_target_model not in self.index_ref_models, "Should not attack the reference models"
        audit_loader = DataLoader(audit_samples, batch_size=self.batch_size, shuffle=False)
        sig_target = self.loss_signal(audit_loader, index_target_model)
        sig_ref_models = []
        for idx in self.index_ref_models:
            sig = self.loss_signal(audit_loader, idx)
            sig_ref_models.append(sig)
        sig_ref_models = torch.stack(sig_ref_models)
        score = -sig_target - torch.logsumexp(-sig_ref_models, dim=0) + np.log(self.prior / (1 - self.prior))
        return score.sigmoid()

class ClassifierAttack:

    def __init__(self, dataset_name, model_type, batch_size, device, index_ref_models, n_loss_samples=20, classifier="MLP"):
        self.dataset = getattr(data, dataset_name)() # Full dataset
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device
        self.index_ref_models = index_ref_models
        self.n_loss_samples = n_loss_samples
        self.classifier = classifier
        self.train_features, self.train_labels, self.val_features, self.val_labels = self.create_attack_dataset()
        if classifier == "MLP":
            self.attack_model = models.MLP(
                in_features=self.train_features.shape[1],
                out_features=1,
                hidden_dims=(128, 256, 128),
            )
            self.attack_model.to(device)
            train_config = utils.Config({
                "batch_size": 8192, # Can probably be very large since data is low dimensional
                "epochs": 500,
                "lr": 1e-3,
                "early_stopping_rounds": 20
            })
            self.train_mlp_classifier(train_config)
        elif classifier == "XGBoost":
            self.attack_model = self.train_xgboost_classifier()
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

    def load_model(self, index):
        model, train_indices  = utils.load_model(
            dataset=self.dataset_name,
            model_type=self.model_type,
            index_model=index,
            device=self.device,
        )
        model.n_rsamples = 1 # Sample multiple loss values as attack features
        return model, train_indices

    @torch.inference_mode()
    def loss_features(self, model, data_samples):
        loss_samples = []
        for _ in range(self.n_loss_samples):
            loss = model.per_sample_loss(data_samples).cpu()
            loss_samples.append(loss)
        loss_samples = torch.stack(loss_samples, dim=1)
        loss_mean = loss_samples.mean(dim=1, keepdim=True)
        loss_std = loss_samples.std(dim=1, keepdim=True)
        loss_features = torch.concat((loss_mean, loss_std, loss_samples.sort(dim=1)[0]), dim=1)
        assert loss_features.shape == (len(data_samples), self.n_loss_samples + 2)
        return loss_features

    @torch.inference_mode()
    def create_attack_dataset(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        features = []
        labels = []
        for index in self.index_ref_models:
            model, train_indices = self.load_model(index)
            train_mask = utils.index_to_mask(train_indices, len(self.dataset))
            running_index = 0
            for samples in dataloader:
                samples = samples.to(self.device)
                n_samples = len(samples)
                indices = torch.arange(running_index, running_index + len(samples))
                running_index += len(samples)
                loss_samples = self.loss_features(model, samples)
                features.append(loss_samples)
                labels.append(train_mask[indices])
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        split_index = int(features.shape[0] * 0.9)
        train_features, train_labels = features[:split_index], labels[:split_index]
        val_features, val_labels = features[split_index:], labels[split_index:]
        return train_features, train_labels, val_features, val_labels

    def train_mlp_classifier(self, config):
        attack_dataset_train = TensorDataset(self.train_features, self.train_labels.to(torch.float))
        attack_dataset_val = TensorDataset(self.val_features, self.val_labels.to(torch.float))
        train_loader = DataLoader(attack_dataset_train, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(attack_dataset_val, batch_size=config.batch_size, shuffle=False)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.attack_model.parameters(), lr=config.lr)
        train_loss = []
        val_loss = []
        min_val_loss = torch.inf
        early_stopping_counter = 0
        for epoch in range(config.epochs):
            self.attack_model.train()
            acc_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.attack_model(X)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                acc_loss += loss.detach().item()
            train_loss.append(acc_loss / len(train_loader))
            self.attack_model.eval()
            with torch.no_grad():
                acc_loss = 0.0
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    out = self.attack_model(X)
                    loss = loss_fn(out, y)
                    acc_loss = loss.detach().item()
                val_loss.append(acc_loss / len(val_loader))
            if val_loss[-1] < min_val_loss:
                min_val_loss = val_loss[-1]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter == config.early_stopping_rounds:
                print(f"Early stopping at epoch={epoch}")
                break

    def train_xgboost_classifier(self):
        X_train, y_train =  self.train_features.numpy(), self.train_labels.numpy()
        X_val, y_val = self.val_features.numpy(), self.val_labels.numpy()
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1000,
            random_state=42,
            early_stopping_rounds=20,
            tree_method="hist",
            device=self.device.type,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model

    @torch.inference_mode()
    def run_attack(self, audit_samples, index_target_model):
        target_model, _ = self.load_model(index_target_model) # don't look at the train_indices here of course
        audit_loader = DataLoader(audit_samples, batch_size=self.batch_size, shuffle=False)
        scores = []
        for samples in audit_loader:
            samples = samples.to(self.device)
            features = self.loss_features(target_model, samples)
            if self.classifier == "XGBoost":
                score = torch.tensor(self.attack_model.predict(features.numpy()))
            else:
                score = self.attack_model(features.to(self.device))
            scores.append(score)
        scores = torch.concat(scores, dim=0).cpu()
        assert scores.shape == (len(audit_samples),)
        return scores
