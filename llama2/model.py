import torch
import torch.nn as nn 
import torch.nn.functional as F 

embed_dim = 4
max_len = 32
vocab_size = 256 
num_heads = 2
num_layers = 4
device = 'cuda' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        self.encoding.requires_grad = False  # We don't want to backprop through the positional encodings

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].to(device)

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x, mask):
        x = x.permute(1, 0, 2)
        attn_x, attn_x_weights = self.attention(
            query=x, 
            key=x, 
            value=x, 
            attn_mask=mask
        )
        x = x.permute(1, 0, 2)
        attn_x = attn_x.permute(1, 0, 2)

        x = attn_x + x # Residual connection
        x = self.attn_norm(x) # RMSNorm
        x = self.dropout(x) # Dropout 

        mlp_x = self.mlp(x)
        x = mlp_x + x # Residual connection 
        x = self.mlp_norm(x) 

        return x 

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([
            Block() for _ in range(num_layers)
        ])

        self.word_emb = nn.Embedding(vocab_size, embed_dim) 
        self.pos_emb = PositionalEncoding()

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        vectors = self.word_emb(x)
        vectors = self.pos_emb(vectors) # NOTE: Here pos_emb already has the residual connection  
        sz = vectors.size(1) # seq_len
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
        for layer in self.layers:
            vectors = layer(vectors, mask=mask) 

        predicted_words = self.fc_out(vectors) 
        return F.softmax(predicted_words, dim=-1)
    
model = Decoder().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(total_params)

with open('input.txt', 'r') as data_file:
    chars = list(data_file.read())
    tokens = [ord(x) for x in chars]

from torch.utils.data import DataLoader, Dataset

class OverlappingSubsequencesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.samples = self.create_samples(data, seq_length)
    
    def create_samples(self, data, seq_length):
        samples = []
        for i in range(len(data) - seq_length):
            samples.append(data[i:i + seq_length + 1])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.tensor(sample[:-1], dtype=torch.long)
        target_seq = torch.tensor(sample[1:], dtype=torch.long)
        return input_seq, target_seq

from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            input_seq, target_seq = batch
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device) 
            
            optimizer.zero_grad()
            
            output = model(input_seq)

            output = output.reshape(-1, output.shape[-1])
            target_seq = target_seq.reshape(-1)
            loss = criterion(output, target_seq)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

src_pad_idx = 0

dataset = OverlappingSubsequencesDataset(tokens, max_len)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

train(model, dataloader, optimizer, criterion, device, num_epochs=10)
