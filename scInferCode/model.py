import torch.nn as nn
import torch.nn.functional as F
import torch


class ContrastiveNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(ContrastiveNetwork, self).__init__()
        self.protein_branch = nn.Sequential(

            nn.Linear(input_size['protein'], embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU()
        )
        self.rna_branch = nn.Sequential(

            nn.Linear(input_size['rna'], hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU()
        )

    def forward(self, protein_input, rna_input):
        protein_embedding = self.protein_branch(protein_input)
        rna_embedding = self.rna_branch(rna_input)
        return protein_embedding, rna_embedding


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        cos_sim = F.cosine_similarity(output1, output2)
        loss = torch.mean((1 - target) * torch.pow(cos_sim, 2) +
                          target * torch.pow(torch.clamp(self.margin - cos_sim, min=0.0), 2))

        return loss


class InferNetwork(nn.Module):
    def __init__(self, input_size, output_size, t=3.0):
        super(InferNetwork, self).__init__()
        self.name = self.__class__.__name__
        self.t = t

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=output_size, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0),  # 更新池化层参数为kernel_size=3

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        )

        length_after_conv_and_pool = input_size

        for _ in range(3):
            length_after_conv_and_pool = (length_after_conv_and_pool - 3) // 3 + 1

        self.fc1 = nn.Linear(128 * length_after_conv_and_pool, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        original_x = x
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        weights = self.fc2(x)
        weights = weights / torch.tensor(self.t, dtype=float)
        weights = F.softmax(weights, dim=1).unsqueeze(2)  # (batch_size, 10, 1)
        weighted_sum = torch.bmm(weights.transpose(1, 2), original_x)  # (batch_size, 1, 300)
        return weighted_sum.squeeze(1)

    def weights_(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        weights = self.fc2(x)
        weights = weights / torch.tensor(self.t, dtype=float)
        weights = F.softmax(weights, dim=1).unsqueeze(2)
        return weights.squeeze(2)


class ClusterLoss(nn.Module):
    def __init__(self):
        super(ClusterLoss, self).__init__()

    def forward(self, x, labels):
        unique_labels = torch.unique(labels)
        num_clusters = len(unique_labels)
        cluster_centers = []

        for label in unique_labels:
            cluster_points = x[labels == label]
            cluster_center = torch.mean(cluster_points, dim=0)
            cluster_centers.append(cluster_center)

        cluster_centers = torch.stack(cluster_centers)

        intra_cluster_distance = 0
        for i, label in enumerate(unique_labels):
            cluster_points = x[labels == label]
            distances = torch.norm(cluster_points - cluster_centers[i], dim=1)
            intra_cluster_distance += torch.sum(distances)

        inter_cluster_distance = 0
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                distance = torch.norm(cluster_centers[i] - cluster_centers[j])
                inter_cluster_distance += distance

        if inter_cluster_distance > 0:
            loss = intra_cluster_distance / inter_cluster_distance
        else:
            loss = intra_cluster_distance

        return loss










