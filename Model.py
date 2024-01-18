import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args

device = "cuda" if t.cuda.is_available() else "cpu"

xavierInit = nn.init.xavier_uniform_
zeroInit = lambda x: nn.init.constant_(x, 0.0)
normalInit = lambda x: nn.init.normal_(x, 0.0, 0.3)

class DSL(nn.Module):
    def __init__(self):
        super(DSL, self).__init__()
        self.uEmbeds0 = nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds0 = nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.LightGCN = LightGCN(self.uEmbeds0, self.iEmbeds0)
        self.LightGCN2 = LightGCN2(self.uEmbeds0)
        self.linear1 = nn.Linear(2*args.latdim, args.latdim)
        self.linear2 = nn.Linear(args.latdim, 1)
        self.dropout = nn.Dropout(args.dropRate)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
        self.sigmoid = nn.Sigmoid()

    def forward(self, adj, uAdj):
        ui_uEmbed0, ui_iEmbed0 = self.LightGCN(adj) # (usr, d)
        uu_Embed0 = self.LightGCN2(uAdj)
        return ui_uEmbed0, ui_iEmbed0, uu_Embed0

    def label(self, lat1, lat2):
        """Calculate labels for user pairs in social graph.

        Args:
            lat1: Embeddings of a batch (1) of user.
            lat2: Embeddings of a batch (2) of user.
        
        Returns:
            Labels for user pairs in soical graph.
        """
        lat = t.cat([lat1, lat2], dim=-1)
        lat = self.leakyrelu(self.dropout(self.linear1(lat))) + lat1 + lat2
        ret = t.reshape(self.sigmoid(self.dropout(self.linear2(lat))), [-1])
        return ret

    def calcLosses(self, adj, usr, itmP, itmN, uAdj, usr0, usrP, usrN, usr1, usr2):
        """Calculate losses for our model.

        Args:
            adj: Adjacency matrix on interaction graph.
            usr: Users sampled on BPRLoss on interaction graph.
            itmP: Positive items sampled on BPRLoss on interaction graph.
            itmN: Negative items sampled on BPRLoss on interaction graph.
            uAdj: Adjacency matrix on social graph.
            usr0: Users sampled on BPRLoss on social graph.
            usrP: Positive users sampled on BPRLoss on social graph.
            usrN: Negative users sampled on BPRLoss on social graph.
            usr1: User pairs (1) sampled on social graph.
            usr2: User pairs (2) sampled on social graph.
        
        Returns:
            Predicting losses on interaction and social graph,
            and self-augmented learning losses.
        """
        ui_uEmbed, ui_iEmbed, uu_Embed = self.forward(adj, uAdj)
        
        pckUlat = ui_uEmbed[usr]
        pckIlatP = ui_iEmbed[itmP]
        pckIlatN = ui_iEmbed[itmN]
        predsP = (pckUlat * pckIlatP).sum(-1)
        predsN = (pckUlat * pckIlatN).sum(-1)
        scoreDiff = predsP - predsN
        preLoss = -(scoreDiff).sigmoid().log().sum() / args.batch # bprloss

        pckUlat = uu_Embed[usr0]
        pckUlatP = uu_Embed[usrP]
        pckUlatN = uu_Embed[usrN]
        predsP = (pckUlat * pckUlatP).sum(-1)
        predsN = (pckUlat * pckUlatN).sum(-1)
        scoreDiff = predsP - predsN
        uuPreLoss = args.uuPre_reg * -(scoreDiff.sigmoid()+1e-8).log().sum() / args.batch # bprloss

        scores = self.label(ui_uEmbed[usr1], ui_uEmbed[usr2])
        preds = (uu_Embed[usr1] * uu_Embed[usr2]).sum(-1)
        salLoss = args.sal_reg * (t.maximum(t.tensor(0.0), 1.0-scores*preds)).sum()
        
        return preLoss, uuPreLoss, salLoss

    def predPairs(self, adj, usr, itm, uAdj):
        """Calculate Similiarity (inner product) of a batch of users and items for evaluating.

        Args:
            adj: Adjacency matrix on interaction graph.
            usr: Users sampled on interaction graph.
            itm: Items sampled on interaction graph.
            uAdj: Adjacency matrix on social graph.
        
        Returns:
            Similiarity of a batch of users and items.
        """
        ret = self.forward(adj, uAdj)
        uEmbeds, iEmbeds = ret[:2]
        uEmbed = uEmbeds[usr]
        iEmbed = iEmbeds[itm]
        return (uEmbed * iEmbed).sum(-1)

class LightGCN(nn.Module):
    def __init__(self, uEmbeds=None, iEmbeds=None, pool='sum'):
        super(LightGCN, self).__init__()
        self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds = iEmbeds if iEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
        self.pool = pool

    def pooling(self, embeds):
        if self.pool == 'mean':
            return embeds.mean(0)
        elif self.pool == 'sum':
            return embeds.sum(0)
        elif self.pool == 'concat':
            return embeds.view(embeds.shape[1], -1)
        else: # final
            return embeds[-1]

    def forward(self, adj):
        embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
        embedLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedLst[-1])
            embedLst.append(embeds)
        embeds = t.stack(embedLst, dim=0)
        embeds = self.pooling(embeds)
        return embeds[:args.user], embeds[args.user:]

class LightGCN2(nn.Module):
    def __init__(self, uEmbeds=None, pool='sum'):
        super(LightGCN2, self).__init__()
        self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.gnnLayers = nn.Sequential(*[GCNLayer() for i in range(args.uugnn_layer)])
        self.pool = pool

    def pooling(self, embeds):
        if self.pool == 'mean':
            return embeds.mean(0)
        elif self.pool == 'sum':
            return embeds.sum(0)
        elif self.pool == 'concat':
            return embeds.view(embeds.shape[1], -1)
        else: # final
            return embeds[-1]

    def forward(self, adj):
        ulats = [self.uEmbeds]
        for gcn in self.gnnLayers:
            temulat = gcn(adj, ulats[-1])
            ulats.append(temulat)
        ulats = t.stack(ulats, dim=0)
        ulats = self.pooling(ulats)
        return ulats

class GCNLayer(nn.Module):
    def __init__(self, edge_dropout=False, msg_dropout=False):
        super(GCNLayer, self).__init__()
        self.edge_dropout = edge_dropout
        self.edge_drop = args.edge_drop
        self.msg_dropout = msg_dropout
        self.msg_drop = args.msg_drop
        self.dropout = nn.Dropout(p=args.dropRate)

    def _sparse_dropout(self, adj, drop_rate):
        keep_rate = 1-drop_rate
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keep_rate).floor()).type(t.bool)
        newVals = vals[mask] / keep_rate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

    def forward(self, adj, embeds):
        if self.edge_dropout:
            adj = self._sparse_dropout(adj, self.edge_drop)
        embeds = t.spmm(adj, embeds)
        if self.msg_dropout:
            embeds = self.dropout(embeds)
        return embeds
