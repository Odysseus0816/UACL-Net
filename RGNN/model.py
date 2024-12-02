import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add


# dynamic adjacency matrix
def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


# 对edge_index和edge_weight将自环移至最后 (node1-node1,node2-node2...)
def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index
    # 不同节点的边
    mask = row != col
    # 对mask的bool值取反 相同节点的自循环
    inv_mask = ~mask
    # 生成的loop_weight一维tensor, 初始值全为1
    loop_weight = torch.full((num_nodes,), fill_value,
                             dtype=None if edge_weight is None else edge_weight.dtype,
                             device=edge_index.device)

    # 将self loop移至edge weight末尾
    if edge_weight is not None:
        # edge_weight中元素的个数=edge_index行的元素个数时正常执行，否则报错
        assert edge_weight.numel() == edge_index.size(1)
        # 相同节点间的edge weight保持不变
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    # 将self loop移至edge index末尾
    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False,
                 bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)

    # allow negative edge weights
    # norm操作等价于计算W、D 然后计算Z
    # 证明https://blog.csdn.net/D_pens/article/details/107879943?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-1.essearch_pc_relevant&spm=1001.2101.3001.4242
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        # x_j: (batch_size*num_nodes*num_nodes, num_features)
        # norm: (batch_size*num_nodes*num_nodes, )
        return norm.view(-1, 1) * x_j


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SymSimGCNNet(torch.nn.Module):
    def __init__(self, num_nodes, initial_edge_weight, num_features, num_hiddens, num_classes=2, K=2, dropout=0.7,
                 learn_edge_weight=True, domain_adaptation="RevGrad"):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            K: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        super(SymSimGCNNet, self).__init__()
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = initial_edge_weight.reshape(self.num_nodes, self.num_nodes)[
            self.xs, self.ys]  # 取edge_weight矩阵的左下三角元素，避免过拟合
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight)  # 将edge weight的左下角元素作为网络可学习的变量
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens, K=K)
        self.fc = nn.Linear(num_hiddens, num_classes)
        if self.domain_adaptation in ["RevGrad"]:
            self.domain_classifier = nn.Linear(num_hiddens, 2)

    def forward(self, data, alpha):
        batch_size = data.batch.max().item() + 1
        x, edge_index = data.x, data.edge_index
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())  # 恢复成对称矩阵进行图卷积
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        # domain classification
        domain_output = None
        if self.domain_adaptation in ["RevGrad"]:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x)
            domain_output = global_add_pool(domain_output, data.batch, size=batch_size)
        x = global_add_pool(x, data.batch, size=batch_size)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas, domain_output


