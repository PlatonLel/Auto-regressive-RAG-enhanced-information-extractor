import torch
import torch.nn.functional as F
# from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor, \
#     BidirectionalEndpointSpanExtractor
from torch import nn

# Helper function to mimic EndpointSpanExtractor
def _endpoint_span_extractor(sequence_tensor: torch.Tensor,
                             span_indices: torch.Tensor,
                             combination: str = "x,y") -> torch.Tensor:
    """
    Extracts spans endpoints like AllenNLP's EndpointSpanExtractor.
    Assumes B x L x K x 2 span_indices format (batch, seq_len, num_spans, 2)
    or B x N x 2 format (batch, num_spans_total, 2).
    Handles indices correctly, clamping them within valid ranges.
    """
    batch_size, seq_len, hidden_dim = sequence_tensor.size()
    
    # Adjust indices shape if needed (common case from model.py)
    # From B x L x K x 2 -> B x (L*K) x 2 
    if span_indices.dim() == 4:
        num_spans_per_token = span_indices.size(2)
        span_indices = span_indices.reshape(batch_size, -1, 2) # B, L*K, 2
    
    num_spans = span_indices.size(1)
    
    # Clamp indices to prevent out-of-bounds errors
    # AllenNLP does this internally
    start_indices = torch.clamp(span_indices[..., 0], 0, seq_len - 1)
    end_indices = torch.clamp(span_indices[..., 1], 0, seq_len - 1)

    # Expand indices for gathering: B x N x D
    start_indices_expanded = start_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    end_indices_expanded = end_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)

    # Gather start and end representations: B x N x D
    start_rep = torch.gather(sequence_tensor, 1, start_indices_expanded)
    end_rep = torch.gather(sequence_tensor, 1, end_indices_expanded)

    if combination == "x,y":
        return torch.cat([start_rep, end_rep], dim=-1)
    elif combination == "x":
        return start_rep
    elif combination == "y":
        return end_rep
    else:
        raise ValueError(f"Invalid combination: {combination}")

# Helper function for attentive span pooling (mean pooling as replacement)
def _mean_pooling_span_extractor(sequence_tensor: torch.Tensor,
                                 span_indices: torch.Tensor) -> torch.Tensor:
    """
    Simple mean pooling over span tokens as a replacement for SelfAttentiveSpanExtractor.
    """
    batch_size, seq_len, hidden_dim = sequence_tensor.size()
    
    if span_indices.dim() == 4:
        span_indices = span_indices.reshape(batch_size, -1, 2) # B, L*K, 2
        
    num_spans = span_indices.size(1)
    span_reps = []

    for i in range(batch_size):
        batch_span_reps = []
        for j in range(num_spans):
            start, end = span_indices[i, j, 0], span_indices[i, j, 1]
            # Clamp indices
            start = torch.clamp(start, 0, seq_len - 1)
            end = torch.clamp(end, 0, seq_len - 1)
            
            # Handle invalid or zero-length spans by returning zeros
            if start > end : # Or should we allow start==end for single tokens? Let's assume valid spans.
                 batch_span_reps.append(torch.zeros(hidden_dim, device=sequence_tensor.device))
                 continue
            
            span_tokens = sequence_tensor[i, start:end + 1, :] # Inclusive end
            batch_span_reps.append(torch.mean(span_tokens, dim=0))
        span_reps.append(torch.stack(batch_span_reps))

    return torch.stack(span_reps) # B x N x D

# Helper function to mimic BidirectionalEndpointSpanExtractor
def _bidir_endpoint_span_extractor(sequence_tensor: torch.Tensor,
                                   span_indices: torch.Tensor) -> torch.Tensor:
    """
    Extracts spans endpoints like AllenNLP's BidirectionalEndpointSpanExtractor.
    Assumes input is concatenated [forward_lstm_output, backward_lstm_output].
    """
    batch_size, seq_len, hidden_dim_x2 = sequence_tensor.size()
    if hidden_dim_x2 % 2 != 0:
         raise ValueError("Input hidden dimension must be even for bidirectional.")
    hidden_dim = hidden_dim_x2 // 2
    
    forward_sequence = sequence_tensor[..., :hidden_dim]
    backward_sequence = sequence_tensor[..., hidden_dim:]

    if span_indices.dim() == 4:
        span_indices = span_indices.reshape(batch_size, -1, 2) # B, L*K, 2

    # Extract endpoints for forward sequence (start_forward, end_forward)
    forward_endpoints = _endpoint_span_extractor(forward_sequence, span_indices, combination="x,y") # B x N x hidden_dim*2
    
    # Extract endpoints for backward sequence (start_backward, end_backward)
    backward_endpoints = _endpoint_span_extractor(backward_sequence, span_indices, combination="x,y") # B x N x hidden_dim*2

    # Concatenate: [start_forward, end_backward, start_backward, end_forward]
    # AllenNLP Bidirectional combines start_forward with end_backward, and start_backward with end_forward
    start_forward = forward_endpoints[..., :hidden_dim]
    end_forward = forward_endpoints[..., hidden_dim:]
    start_backward = backward_endpoints[..., :hidden_dim]
    end_backward = backward_endpoints[..., hidden_dim:]
    
    # The specific combination might vary based on AllenNLP version/config, 
    # common one is start_forward + end_backward and end_forward + start_backward
    # Here we just concat all 4 parts for flexibility in downstream projection
    return torch.cat([start_forward, end_backward, start_backward, end_forward], dim=-1) # B x N x hidden_dim*4
    

class SpanQuery(nn.Module):

    def __init__(self, hidden_size, max_width, trainable=True):
        super().__init__()

        self.query_seg = nn.Parameter(torch.randn(hidden_size, max_width))

        nn.init.uniform_(self.query_seg, a=-1, b=1)

        if not trainable:
            self.query_seg.requires_grad = False

        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, *args):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        span_rep = torch.einsum('bld, ds->blsd', h, self.query_seg)

        return self.project(span_rep)


class SpanMLP(nn.Module):

    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.mlp = nn.Linear(hidden_size, hidden_size * max_width)

    def forward(self, h, *args):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        span_rep = self.mlp(h)

        span_rep = span_rep.view(B, L, -1, D)

        return span_rep.relu()


class SpanCAT(nn.Module):

    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.max_width = max_width

        self.query_seg = nn.Parameter(torch.randn(128, max_width))

        self.project = nn.Sequential(
            nn.Linear(hidden_size + 128, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, *args):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        h = h.view(B, L, 1, D).repeat(1, 1, self.max_width, 1)

        q = self.query_seg.view(1, 1, self.max_width, -1).repeat(B, L, 1, 1)

        span_rep = torch.cat([h, q], dim=-1)

        span_rep = self.project(span_rep)

        return span_rep


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class SpanEndpoints(nn.Module):

    def __init__(self, hidden_size, max_width, width_embedding=128):
        super().__init__()
        # self.span_extractor = EndpointSpanExtractor(hidden_size,
        #                                             combination='x,y')
        # No specific extractor module needed, logic is in forward

        self.downproject = nn.Sequential(
            Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        span_rep = _endpoint_span_extractor(h, span_idx, combination="x,y") # B x N x D*2

        return self.downproject(span_rep).view(B, L, -1, D)


class SpanAttention(nn.Module):
    # Note: Replacing SelfAttentiveSpanExtractor with Mean Pooling
    def __init__(self, hidden_size, max_width, width_embedding=128):
        super().__init__()
        # self.span_extractor = SelfAttentiveSpanExtractor(hidden_size,
        #                                                  num_width_embeddings=max_width,
        #                                                  span_width_embedding_dim=width_embedding,
        #                                                  )
        # self.downproject = nn.Sequential(
        #     nn.Linear(hidden_size + width_embedding, hidden_size),
        #     nn.ReLU()
        # )
        # No specific extractor module needed, logic is in forward
        # Downproject adjusted for mean pooling output dimension
        self.downproject = nn.Sequential(
             Linear(hidden_size, hidden_size), # Input dim is hidden_size from mean pooling
             nn.ReLU()
        )

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()
        # span_rep = self.span_extractor(h, span_idx)
        span_rep = _mean_pooling_span_extractor(h, span_idx) # B x N x D

        return self.downproject(span_rep).view(B, L, -1, D)


class Bidir(nn.Module):

    def __init__(self, hidden_size, max_width):
        super().__init__()
        # self.span_extractor = BidirectionalEndpointSpanExtractor(hidden_size)
        # Logic is in forward. Output dim of helper is hidden_size * 4
        # Adjust downproject layer input dimension accordingly
        self._bidir_output_dim = hidden_size * 4 

        self.downproject = nn.Sequential(
            # Linear(self.span_extractor.get_output_dim(), hidden_size),
            Linear(self._bidir_output_dim, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size() # D here is hidden_size * 2 from BiLSTM
        # span_rep = self.span_extractor(h, span_idx)
        span_rep = _bidir_endpoint_span_extractor(h, span_idx) # B x N x (D/2)*4 = B x N x D*2

        return self.downproject(span_rep).view(B, L, -1, D // 2) # Reshape back to original hidden_size


class SpanConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, span_mode='conv_normal'):
        super().__init__()

        if span_mode == 'conv_conv':
            self.conv = nn.Conv1d(hidden_size, hidden_size,
                                  kernel_size=kernel_size)
        elif span_mode == 'conv_max':
            self.conv = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        elif span_mode == 'conv_mean' or span_mode == 'conv_sum':
            self.conv = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

        self.span_mode = span_mode

        self.pad = kernel_size - 1

    def forward(self, x):

        x = torch.einsum('bld->bdl', x)

        if self.pad > 0:
            x = F.pad(x, (0, self.pad), "constant", 0)

        x = self.conv(x)

        if self.span_mode == "conv_sum":
            x = x * (self.pad + 1)

        return torch.einsum('bdl->bld', x)


class SpanConv(nn.Module):
    def __init__(self, hidden_size, max_width, span_mode):
        super().__init__()

        kernels = [i + 2 for i in range(max_width - 1)]

        self.convs = nn.ModuleList()

        for kernel in kernels:
            self.convs.append(SpanConvBlock(hidden_size, kernel, span_mode))

        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x, *args):

        span_reps = [x]

        for conv in self.convs:
            h = conv(x)
            span_reps.append(h)

        span_reps = torch.stack(span_reps, dim=-2)

        return self.project(span_reps)


class SpanEndpointsBlock(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x):
        B, L, D = x.size()

        span_idx = torch.LongTensor(
            [[i, i + self.kernel_size - 1] for i in range(L)]).to(x.device)

        x = F.pad(x, (0, 0, 0, self.kernel_size - 1), "constant", 0)

        # endrep
        start_end_rep = torch.index_select(x, dim=1, index=span_idx.view(-1))

        start_end_rep = start_end_rep.view(B, L, 2, D)

        return start_end_rep


class SpanEndpointsV2(nn.Module):
    def __init__(self, hidden_size, max_width, span_mode='endpoints_mean'):
        super().__init__()

        assert span_mode in ['endpoints_mean',
                             'endpoints_max', 'endpoints_cat']

        self.K = max_width

        kernels = [i + 1 for i in range(max_width)]

        self.convs = nn.ModuleList()

        for kernel in kernels:
            self.convs.append(SpanEndpointsBlock(kernel))

        self.span_mode = span_mode

    def forward(self, x, *args):

        span_reps = []

        for conv in self.convs:
            h = conv(x)

            span_reps.append(h)

        span_reps = torch.stack(span_reps, dim=-3)

        if self.span_mode == 'endpoints_mean':
            span_reps = torch.mean(span_reps, dim=-2)
        elif self.span_mode == 'endpoints_max':
            span_reps = torch.max(span_reps, dim=-2).values
        elif self.span_mode == 'endpoints_cat':
            span_reps = span_reps.view(B, L, self.K, -1)

        return span_reps


class ConvShare(nn.Module):
    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.max_width = max_width

        self.conv_weigth = nn.Parameter(
            torch.randn(hidden_size, hidden_size, max_width))

        nn.init.xavier_normal_(self.conv_weigth)

    def forward(self, x, *args):
        span_reps = []

        x = torch.einsum('bld->bdl', x)

        for i in range(self.max_width):
            pad = i
            x_i = F.pad(x, (0, pad), "constant", 0)
            conv_w = self.conv_weigth[:, :, :i + 1]
            out_i = F.conv1d(x_i, conv_w)
            span_reps.append(out_i.transpose(-1, -2))

        return torch.stack(span_reps, dim=-2)


class ConvShareEndpoints(nn.Module):
    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.max_width = max_width
        self.out_size = hidden_size * 3 # This might need adjustment depending on proj layers
        # self.span_extractor = EndpointSpanExtractor(hidden_size,
        #                                             combination='x,y')
        # Logic in forward
        self.conv_share = ConvShare(hidden_size, max_width)

        self.out_project_end = nn.Linear(hidden_size * 2, hidden_size // 2)
        self.out_project_conv = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, h, span_idx):
        B, L, D = h.size()
        # span_rep_end = self.span_extractor(h, span_idx)
        span_rep_end = _endpoint_span_extractor(h, span_idx, combination="x,y") # B x N x D*2

        # Reshape span_idx for conv_share if needed (should be B x L x K x 2)
        # The conv_share forward doesn't take span_idx, so it's okay.
        
        # Apply projection and reshape endpoint reps
        span_rep_end = self.out_project_end(span_rep_end).view(B, L, self.max_width, -1)

        span_rep_conv = self.conv_share(h) # Output: B x L x K x D
        span_rep_conv = self.out_project_conv(span_rep_conv) # Output: B x L x K x D//2 (already reshaped)

        return torch.cat([span_rep_end, span_rep_conv], dim=-1)


class SpanMarker(nn.Module):

    def __init__(self, hidden_size, max_width, dropout=0.4):
        super().__init__()

        self.max_width = max_width

        self.project_start = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )

        self.project_end = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )

        # self.span_extractor_start = EndpointSpanExtractor(hidden_size,
        #                                                   combination='x')

        # self.span_extractor_end = EndpointSpanExtractor(hidden_size,
        #                                                 combination='y')
        # Logic in forward

        self.out_project = nn.Linear(hidden_size * 2, hidden_size, bias=True)

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        # project start and end
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        # extract span
        # start_span_rep = self.span_extractor_start(start_rep, span_idx)
        # end_span_rep = self.span_extractor_end(end_rep, span_idx)
        start_span_rep = _endpoint_span_extractor(start_rep, span_idx, combination="x")
        end_span_rep = _endpoint_span_extractor(end_rep, span_idx, combination="y")

        # concat start and end
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()

        # project
        cat = self.out_project(cat)

        # reshape
        return cat.view(B, L, self.max_width, D)


class SpanMarkConv(nn.Module):
    def __init__(self, hidden_size, max_width, dropout=0.4):
        super().__init__()

        self.max_width = max_width

        self.project = nn.Linear(hidden_size, hidden_size * 2)
        # self.span_extractor_start = EndpointSpanExtractor(hidden_size,
        #                                                   combination='x')

        # self.span_extractor_end = EndpointSpanExtractor(hidden_size,
        #                                                 combination='y')
        # Logic in forward

        self.conv_share = ConvShare(hidden_size, max_width)

        self.out_project = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        # project start and end
        start_rep, end_rep = self.project(h).chunk(2, dim=-1)

        # extract span
        # start_span_rep, end_span_rep = self.span_extractor_start(start_rep, span_idx), \
        #     self.span_extractor_end(end_rep, span_idx)
        start_span_rep = _endpoint_span_extractor(start_rep, span_idx, combination="x")
        end_span_rep = _endpoint_span_extractor(end_rep, span_idx, combination="y")

        conv_span_rep = self.conv_share(h).reshape(B, -1, D)  # conv feature # B x (L*K) x D

        # Need to align dimensions before concatenating
        # Endpoint extractors output B x N x D, conv_share B x (L*K) x D
        # Assuming N = L*K
        
        # concat start and end
        cat = torch.cat([start_span_rep, end_span_rep, conv_span_rep], dim=-1) # B x N x (D+D+D)

        # project
        cat = self.out_project(cat) # Adjust input dim of Linear in out_project if needed

        return cat.view(B, L, self.max_width, D)


class SpanRepLayer(nn.Module):
    """
    Various span representation approaches
    """

    def __init__(self, hidden_size, max_width, span_mode, p_drop=0.4):
        super().__init__()

        if span_mode == 'endpoints':
            self.span_rep_layer = SpanEndpoints(hidden_size, max_width)
        elif span_mode == 'attentive':
            self.span_rep_layer = SpanAttention(hidden_size, max_width)
        elif span_mode == 'marker':
            self.span_rep_layer = SpanMarker(hidden_size, max_width, p_drop)
        elif span_mode == 'markconv':
            self.span_rep_layer = SpanMarkConv(hidden_size, max_width)
        elif span_mode == 'birectionnal':
            self.span_rep_layer = Bidir(hidden_size, max_width)
        elif span_mode == 'query':
            self.span_rep_layer = SpanQuery(
                hidden_size, max_width, trainable=True)
        elif span_mode == 'mlp':
            self.span_rep_layer = SpanMLP(hidden_size, max_width)
        elif span_mode == 'cat':
            self.span_rep_layer = SpanCAT(hidden_size, max_width)
        elif span_mode == 'conv_conv':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_conv')
        elif span_mode == 'conv_max':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_max')
        elif span_mode == 'conv_mean':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_mean')
        elif span_mode == 'conv_sum':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_sum')
        elif span_mode == 'conv_share':
            self.span_rep_layer = ConvShare(hidden_size, max_width)
        elif span_mode == 'conv_share_endpoints':
            self.span_rep_layer = ConvShareEndpoints(hidden_size, max_width)
        else:
            self.span_rep_layer = SpanEndpointsV2(
                hidden_size, max_width, span_mode=span_mode)

    def forward(self, x, *args):
        return self.span_rep_layer(x, *args)
