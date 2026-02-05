import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GINConv as GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, Batch
from model.dataset import embedding_dim, hidden_size

class MPFIL_MutPred(nn.Module):
    def __init__(self):
        super(MPFIL, self).__init__()
        self.gcn_1 = GCNConv(nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.SiLU()), eps=1e-06, train_eps=True)
        self.gcn_2 = GCNConv(nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 2), nn.SiLU()), eps=1e-06, train_eps=True)
        self.gcn_3 = GCNConv(nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.SiLU()), eps=1e-06, train_eps=True)
        self.gcn_res = GCNConv(nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.SiLU()), eps=1e-06, train_eps=True)
        self.fc_post = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, embedding_dim))

        self.graph_pair_att = nn.MultiheadAttention(embed_dim=2, num_heads=1, batch_first=True)

        self.fc_graph_1 = nn.Linear(embedding_dim + embedding_dim, embedding_dim)
        self.fc_graph_2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc_res = nn.Linear(embedding_dim, embedding_dim)
        self.norm_1 = nn.LayerNorm(embedding_dim, eps=1e-06)
        self.norm_2 = nn.LayerNorm(embedding_dim, eps=1e-06)

        self.dropout_gcn, self.dropout_gcn_2 = nn.Dropout(p=0.1), nn.Dropout(p=0.2)

        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=hidden_size, dropout=0.1, batch_first=True)  # nhead=2,4,8
        self.contextual_encoder = TransformerEncoder(encoder_layer, num_layers=1)  # num_layers=1,2,3,....
        # self.contextual_encoder = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)   # If there is insufficient video memory, BiGRU can be used as an alternative.
        self.transform_query = nn.Linear(embedding_dim, 1, bias=False)
        self.pairwise_extraction_context = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=96, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU()
        )


        self.residue_encoder = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=80, kernel_size=49, stride=1, padding=30),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(in_channels=80, out_channels=40, kernel_size=31, stride=1, padding=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(in_channels=40, out_channels=16, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        self.pairwise_extraction_residue = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=96, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.pooler = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        self.relu = nn.ReLU()

        self.fc_AB = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim * 2, int(embedding_dim / 2)),
            nn.BatchNorm1d(int(embedding_dim / 2)),
            nn.ReLU()
        )

        self.protein_level_att = nn.MultiheadAttention(embed_dim=2, num_heads=1, batch_first=True)
        self.complex_pair_att = nn.MultiheadAttention(embed_dim=2, num_heads=1, batch_first=True)

        self.fc_final = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim*2 + embedding_dim*2 + embedding_dim + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, data, length, edge):
        graph_format_data = []
        ''' Construct Protein Graph '''
        for i in range(data.shape[0]):
            edge_values = edge.x[torch.where(edge.batch == i)].T
            cur = Data(x=data[i][1][1: length[i]+1], y=data[i][2][1: length[i]+1], edge_index=edge_values)
            graph_format_data.append(cur)
        graph_protein = Batch.from_data_list(graph_format_data)
        G_wild, G_mut = graph_protein.x, graph_protein.y
        edge_index, batch = graph_protein.edge_index, graph_protein.batch

        ''' Graph-level Structural Feature Encoding '''
        residual, residual_index = G_wild, edge_index
        G_wild_1 = self.gcn_1(G_wild, edge_index)
        G_wild_1 = F.silu(G_wild_1)
        G_wild_1 = self.dropout_gcn(G_wild_1)

        G_wild_2 = self.gcn_2(G_wild_1, edge_index)
        G_wild_2 = F.silu(G_wild_2)
        G_wild_2 = self.dropout_gcn(G_wild_2)

        G_wild_3 = self.gcn_3(G_wild_2, edge_index)
        G_wild_3 = G_wild_3 + self.fc_post(self.gcn_res(residual, residual_index))
        G_wild_3 = F.silu(G_wild_3)
        G_wild_3 = self.dropout_gcn_2(G_wild_3)

        residual, residual_index = G_mut, edge_index
        G_mut_1 = self.gcn_1(G_mut, edge_index)
        G_mut_1 = F.silu(G_mut_1)
        G_mut_1 = self.dropout_gcn(G_mut_1)

        G_mut_2 = self.gcn_2(G_mut_1, edge_index)
        G_mut_2 = F.silu(G_mut_2)
        G_mut_2 = self.dropout_gcn(G_mut_2)

        G_mut_3 = self.gcn_3(G_mut_2, edge_index)
        G_mut_3 = G_mut_3 + self.fc_post(self.gcn_res(residual, residual_index))
        G_mut_3 = F.silu(G_mut_3)
        G_mut_3 = self.dropout_gcn_2(G_mut_3)

        G_wild_4, G_mut_4 = global_mean_pool(G_wild_3, batch), global_mean_pool(G_mut_3, batch)
        G_wm = torch.stack((G_wild_4, G_mut_4), dim=2)
        G_wm = G_wm / torch.mean(self.relu(G_wm), dim=1, keepdim=True)
        G_wm = self.graph_pair_att(G_wm, G_wm, G_wm, need_weights=False)[0]
        gnn_out = self.fc_graph_1(G_wm.reshape(-1, embedding_dim*2))
        residual = self.fc_res(G_wild_3 - G_mut_3)
        residual = global_mean_pool(residual, batch)
        gnn_out = self.norm_1(gnn_out) + self.dropout_gcn(self.norm_2(residual))
        gnn_out = F.silu(gnn_out)
        gnn_out = self.fc_graph_2(gnn_out)

        ''' Multi-perspective Sequential Feature Encoding '''
        ### Global level Differential Feature
        data = data.view(-1, 500, embedding_dim)  # (batch_size*3, 500, embedding_dim)
        CLS = data[:, 0].squeeze()  # (batch_size*3, embedding_dim)
        CLS = self.pooler(CLS)      # (batch_size*3, embedding_dim)
        CLS = CLS.view(-1, 3, embedding_dim)  # (batch_size, 3, embedding_dim)
        CLS = CLS / torch.mean(self.relu(CLS), dim=2, keepdim=True) * 2
        A, B, B_m = CLS[:, 0], CLS[:, 1], CLS[:, 2]   # (batch_size, embedding_dim)
        AB__AB_m = torch.stack((torch.stack((A, B), dim=2), torch.stack((A, B_m), dim=2)), dim=1).view(-1, embedding_dim*2, 2)  # (batch_size, 2*embedding_dim, 2)
        AB__AB_m = self.protein_level_att(AB__AB_m, AB__AB_m, AB__AB_m, need_weights=False)[0]
        AB__AB_m = AB__AB_m.reshape(-1, embedding_dim*2)  # (batch_size*2, 2*embedding_dim)
        AB__AB_m = self.fc_AB(AB__AB_m)  # (batch_size*2, embedding_dim/2)
        PQ = AB__AB_m.view(-1, 2, int(embedding_dim/2))  # (batch_size, 2, embedding_dim/2)
        PQ = PQ.permute(0, 2, 1)  # (batch_size, embedding_dim/2 , 2)
        PQ = PQ / torch.mean(self.relu(PQ), dim=1, keepdim=True)   # (batch_size, embedding_dim/2 , 2)
        global_out = self.complex_pair_att(PQ, PQ, PQ, need_weights=False)[0]  # (batch_size, embedding_dim/2,  2)
        global_out = global_out.reshape(-1, embedding_dim)  # (batch_size, embedding_dim)

        ### Full-sequence Context-level Feature
        transform_out = self.contextual_encoder(data)  # [batch*3, 500, embedding_dim]
        A = self.transform_query(transform_out)
        A = torch.softmax(A, 1)
        transform_out = torch.sum(transform_out * A, 1)      # [batch*3, embedding_dim]
        extract_in = transform_out.view(-1, 3, 16, 64)   # [batch, 3, 16, 64]
        extract_in = extract_in.permute(1, 0, 3, 2)  # [3, batch, 64, 16]
        extract_in = torch.stack([torch.cat([extract_in[0], extract_in[1]], dim=1),
                                  torch.cat([extract_in[0], extract_in[2]], dim=1)], dim=1)
        extract_in = extract_in.view(-1, 128, 16)
        context_out = self.pairwise_extraction_context(extract_in)

        ### Residue-level Sequential Feature
        encoder_out = self.residue_encoder(data.permute(0, 2, 1))  # output_dim --> [batch*3, 16, 64]
        extract_in = encoder_out.view(-1, 3, 16, 64)  # [batch, 3, 16, 64]
        extract_in = extract_in.permute(1, 0, 3, 2)   # [3, batch, 64, 16]
        extract_in = torch.stack([torch.cat([extract_in[0], extract_in[1]], dim=1),
                                     torch.cat([extract_in[0], extract_in[2]], dim=1)], dim=1)
        extract_in = extract_in.view(-1, 128, 16)  # [batch*2, 128 ,16]
        residue_out = self.pairwise_extraction_residue(extract_in)  # [batch*2, 64, 16]

        ''' Final Prediction Layer '''
        fc_in = torch.cat((residue_out.view(-1, embedding_dim*2), context_out.view(-1, embedding_dim*2), global_out, gnn_out), dim=1)  # [batch, embedding_dim*6]
        fc_out = self.fc_final(fc_in)

        return fc_out