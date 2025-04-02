import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer


#  get_output_dim()

class TokenRep(nn.Module):

    def __init__(self, num_queries=40, model_name="bert-base-cased", fine_tune=True, subtoken_pooling="first"):

        super().__init__()

        # Initialize tokenizer and model directly from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Set fine-tuning mode
        if not fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.encoder.config.hidden_size

        self.num_queries = num_queries

        self.subtoken_pooling = subtoken_pooling

        # Initialize query embeddings if needed
        if self.num_queries == 0:
            self.query_embedding = None
        else:
            e_size = self.encoder.embeddings.word_embeddings.embedding_dim
            self.query_embedding = nn.Parameter(torch.randn(num_queries, e_size))
            nn.init.uniform_(self.query_embedding, -0.01, 0.01)

    def _tokenize_and_align(self, tokens):
        """
        Tokenize text and keep track of token to subtoken alignment
        """
        batch_tokens = []
        batch_subtoken_lengths = []
        
        for sentence_tokens in tokens:
            # Tokenize each token separately and keep track of lengths
            subtokens = []
            subtoken_lengths = []
            
            for token in sentence_tokens:
                token_subtokens = self.tokenizer.tokenize(token)
                subtokens.extend(token_subtokens)
                subtoken_lengths.append(len(token_subtokens))
            
            batch_tokens.append(subtokens)
            batch_subtoken_lengths.append(subtoken_lengths)
        
        # Add special tokens and pad
        encoded = self.tokenizer(
            batch_tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return encoded, batch_subtoken_lengths

    def _pool_subtokens(self, hidden_states, subtoken_lengths, batch_size):
        """
        Pool subtokens according to the specified strategy
        """
        device = hidden_states.device
        max_tokens = max(sum(lengths) for lengths in subtoken_lengths)
        pooled = torch.zeros(batch_size, max_tokens, self.hidden_size, device=device)
        
        offset = 1  # Account for [CLS] token
        for b in range(batch_size):
            for i, length in enumerate(subtoken_lengths[b]):
                if length == 0:
                    continue
                    
                subtokens = hidden_states[b, offset:offset+length]
                
                if self.subtoken_pooling == "first":
                    pooled_value = subtokens[0]
                elif self.subtoken_pooling == "last":
                    pooled_value = subtokens[-1]
                elif self.subtoken_pooling == "mean":
                    pooled_value = subtokens.mean(dim=0)
                else:  # default to first
                    pooled_value = subtokens[0]
                    
                pooled[b, i] = pooled_value
                offset += length
                
            offset = 1  # Reset for next batch item
            
        return pooled

    def forward(self, tokens, lengths):
        # Get the device
        device = next(self.parameters()).device
        
        # Tokenize and align
        encoded, subtoken_lengths = self._tokenize_and_align(tokens)
        
        # Move inputs to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Prepare query embeddings if needed
        if self.query_embedding is not None:
            batch_size = input_ids.size(0)
            queries = self.query_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Concatenate query embeddings with input embeddings
            input_embeddings = self.encoder.embeddings.word_embeddings(input_ids)
            input_embeddings = torch.cat([queries, input_embeddings], dim=1)
            
            # Adjust attention mask for queries
            query_mask = torch.ones(batch_size, self.num_queries, device=device)
            attention_mask = torch.cat([query_mask, attention_mask], dim=1)
            
            # Get model outputs
            outputs = self.encoder(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Split queries and token representations
            hidden_states = outputs.hidden_states[-1]
            queries = hidden_states[:, :self.num_queries]
            hidden_states = hidden_states[:, self.num_queries:]
            
        else:
            # Get model outputs without queries
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            queries = None
        
        # Pool subtokens
        pooled = self._pool_subtokens(hidden_states, subtoken_lengths, input_ids.size(0))
        
        # Create attention mask for pooled tokens
        B = len(lengths)
        max_length = lengths.max()
        mask = (torch.arange(max_length, device=device).view(1, -1).expand(B, -1) < lengths.unsqueeze(1))
        
        # Prepare cache for transformer decoder
        if self.query_embedding is not None:
            actual_memory_pad_mask = ~attention_mask[:, self.num_queries:].bool()
        else:
            actual_memory_pad_mask = ~attention_mask.bool()
            
        cache = {
            "memory": hidden_states,
            "memory_pad_mask": actual_memory_pad_mask
        }
        
        return {
            "embeddings": pooled,
            "mask": mask,
            "queries": queries,
            "cache": cache
        }

#
# # test with main
# if __name__ == '__main__':
#     model = TokenRep(num_queries=3, model_name="bert-base-uncased")
#
#     tokens = ["This is a test", "This is another testhjfkf fhjzfhryb"]
#     lengths = torch.tensor([len(i.split()) for i in tokens])
#
#     tokens = [i.split() for i in tokens]
#
#     out = model(tokens, lengths)
#
#     print(out["embeddings"].shape)
#     print(out["original"][0].shape)
#     print(out["original"][1].shape)
#     print(out["original"][1])
