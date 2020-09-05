import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation must keep the contracts (method name,
# arguments and return values) and attributes as given
# for each model because that is what the unit tests your
# code will be automatically graded with expect.
#
# Using nn.Module and "forward" tells torch which
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention.


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class RNN(nn.Module):
    """ A stacked vanilla RNN with Tanh nonlinearities."""
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size,
                 num_layers, dp_keep_prob):
        """
        Initialization of the parameters of the recurrent and fc layers.
        Supports any number of stacked hidden layers (specified by num_layers),
        uses an input embedding layer, and includes fully connected layers with
        dropout after each recurrent layer.

        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Dropout is not applied on the recurrent connections.
        """
        super(RNN, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dp_keep_prob = dp_keep_prob
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(self.vocab_size,self.emb_size)

        # Create layers
        self.layers = nn.ModuleList()
        # The first layer
        self.layers.append(nn.Linear(emb_size + hidden_size, hidden_size))
        # The hidden layers
        self.layers.extend(clones(nn.Linear(2*hidden_size, hidden_size), num_layers-1))
        # Dropout
        self.dropout = nn.Dropout(1 - self.dp_keep_prob)
        # The output layer
        self.out_layer = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize the embedding and output weights uniformly."""
        # Intialize embedding weights unformly in the range [-0.1, 0.1]
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        # For every layer
        for i in range(self.num_layers):
            # Initialize the weights and biases uniformly
            b = 1/math.sqrt(self.hidden_size)
            nn.init.uniform_(self.layers[i].weight, -b, b)
            nn.init.uniform_(self.layers[i].bias, -b, b)
        # Initialize output layer weights uniformly in the range [-0.1, 0.1]
        # And all the biases to 0
        nn.init.uniform_(self.out_layer.weight, -0.1, 0.1)
        nn.init.zeros_(self.out_layer.bias)

    def init_hidden(self):
        """Initialize the hidden states to zero.

        This is used for the first mini-batch in an epoch, only.
        """
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        """ Compute the recurrent updates.

        Compute the forward pass, using nested python for loops.
        The outer for loop iterates over timesteps, and the inner for loop iterates
        over hidden layers of the stack.

        Within these for loops, the parameter tensors and nn.modules
        created in __init__ are used to compute the recurrent updates according to
        the equations provided in the .tex of the assignment.

        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in run_exp.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                        shape: (num_layers, batch_size, hidden_size)
        """
        if inputs.is_cuda:
            device = inputs.get_device()
        else:
            device = torch.device("cpu")

        # Apply the Embedding layer on the input
        embed_out = self.embeddings(inputs)# shape (seq_len,batch_size,emb_size)

        # Create a tensor to store outputs during the Forward
        logits = torch.zeros(self.seq_len, self.batch_size, self.vocab_size).to(device)

        # For each time step
        for timestep in range(self.seq_len):
            # Apply dropout on the embedding result
            input_ = self.dropout(embed_out[timestep])
            # For each layer
            for layer in range(self.num_layers):
                # Calculate the hidden states
                # And apply the activation function tanh on it
                hidden[layer] = torch.tanh(self.layers[layer](torch.cat([input_, hidden[layer]], 1)))
                # Apply dropout on this layer, but not for the recurrent units
                input_ = self.dropout(hidden[layer])
            # Store the output of the time step
            logits[timestep] = self.out_layer(input_)

        return logits, hidden

    # Problem 4.2
    def generate(self, inputs, hidden, generated_seq_len):
        """
        Generate a sample sequence from the RNN.

        This is similar to the forward method but instead of having ground
        truth input for each time step, you are now required to sample the token
        with maximum probability at each time step and feed it as input at the
        next time step.

        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        if inputs.is_cuda:
            device = inputs.get_device()
        else:
            device = torch.device("cpu")

        # Create a tensor to store outputs during the Forward
        logits = torch.zeros(generated_seq_len, self.batch_size).to(device)

        # For each time step
        for timestep in range(generated_seq_len):
            # Apply dropout on the embedding result
            input_ = self.dropout(self.embeddings(inputs))
            # For each layer
            for layer in range(self.num_layers):
                # Calculate the hidden states
                # And apply the activation function tanh on it
                hidden[layer] = torch.tanh(self.layers[layer](torch.cat([input_, hidden[layer]], 1)))
                # Apply dropout on this layer, but not for the recurrent units
                input_ = self.dropout(hidden[layer])
            
            inputs = torch.argmax(torch.softmax(self.out_layer(input_),1), dim=1)    
            # Store the output of the time step
            logits[timestep] = inputs

        return logits


# Problem 1
class GRU(nn.Module): # Implement a stacked GRU RNN
    """A stacked gated recurrent unit (GRU) RNN.

    Follow the same template as the RNN (above), but use the equations for
    GRU, not Vanilla RNN.

    Use the attribute names that are provided.

    Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    and output biases to 0 (in place). The embeddings should not use a bias vector.
    Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
    in the range [-k, k] where k is the square root of 1/hidden_size

    IMPORTANT: For each init method, use a call to nn.init once for the weights
    and once for the biases, in that order. If you follow the wrong order or
    call nn.init a different number of times the Gradescope tests will fail.
    """
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size,
               num_layers, dp_keep_prob):
        super(GRU, self).__init__()
        # Model parameters
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.batch_size = batch_size
        # TODO ========================

        self.word_embeddings = nn.Embedding(self.vocab_size, self.emb_size)

        # Create "reset gate" layers
        self.r = nn.ModuleList()
        self.r.append(nn.Linear(emb_size + hidden_size, hidden_size))
        self.r.extend(clones(nn.Linear(2*hidden_size, hidden_size), num_layers-1))

        # Create "forget gate" layers
        self.z = nn.ModuleList()
        self.z.append(nn.Linear(emb_size + hidden_size, hidden_size))
        self.z.extend(clones(nn.Linear(2*hidden_size, hidden_size), num_layers-1))

        # Create the "memory content" layers
        self.h = nn.ModuleList()
        self.h.append(nn.Linear(emb_size + hidden_size, hidden_size))
        self.h.extend(clones(nn.Linear(2*hidden_size, hidden_size), num_layers-1))

        # Dropout
        self.dropout = nn.Dropout(1 - self.dp_keep_prob)

        # The output layer
        self.out_layer = nn.Linear(hidden_size, vocab_size)

        self.init_embedding_weights_uniform()
        self.init_reset_gate_weights_uniform()      
        self.init_forget_gate_weights_uniform()
        self.init_memory_weights_uniform()
        self.init_out_layer_weights_uniform()

    def init_embedding_weights_uniform(self, init_range=0.1):
        nn.init.uniform_(self.word_embeddings.weight, -0.1, 0.1)

    def init_reset_gate_weights_uniform(self):
        for i in range(self.num_layers):
            # Initialize the weights and biases uniformly
            b = 1/math.sqrt(self.hidden_size)
            nn.init.uniform_(self.r[i].weight, -b, b)
            nn.init.uniform_(self.r[i].bias, -b, b)

    def init_forget_gate_weights_uniform(self):
        for i in range(self.num_layers):
            # Initialize the weights and biases uniformly
            b = 1/math.sqrt(self.hidden_size)
            nn.init.uniform_(self.z[i].weight, -b, b)
            nn.init.uniform_(self.z[i].bias, -b, b)

    def init_memory_weights_uniform(self):
        for i in range(self.num_layers):
            # Initialize the weights and biases uniformly
            b = 1/math.sqrt(self.hidden_size)
            nn.init.uniform_(self.h[i].weight, -b, b)
            nn.init.uniform_(self.h[i].bias, -b, b)

    def init_out_layer_weights_uniform(self):
        nn.init.uniform_(self.out_layer.weight, -0.1, 0.1)
        nn.init.zeros_(self.out_layer.bias)

    def init_hidden(self):
        """
        This method returns a tensor of shape
        (self.num_layers, self.batch_size, self.hidden_size)
        filled with zeros as the initial hidden states of the GRU.
        """
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        """ Compute the recurrent updates.

        Compute the forward pass, using nested python for loops.
        The outer for loop iterates over timesteps, and the inner for loop iterates
        over hidden layers of the stack.

        Within these for loops, the parameter tensors and nn.modules
        created in __init__ are used to compute the recurrent updates according to
        the equations provided in the .tex of the assignment.

        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in run_exp.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                        shape: (num_layers, batch_size, hidden_size)
        """
        if inputs.is_cuda:
            device = inputs.get_device()
        else:
            device = torch.device("cpu")

        # Apply the Embedding layer on the input
        embed_out = self.word_embeddings(inputs)# shape (seq_len,batch_size,emb_size)

        # Create a tensor to store outputs during the Forward
        logits = torch.zeros(self.seq_len, self.batch_size, self.vocab_size).to(device)

        # For each time step
        for timestep in range(self.seq_len):
            # Apply dropout on the embedding result
            input_ = self.dropout(embed_out[timestep])
            # For each layer
            # Creating this because autograd doesnt work if replaced inplace.
            temp_hidden = []
            for layer in range(self.num_layers):
                # Calculate the hidden states
                # And apply the activation function tanh on it
                r = torch.sigmoid(self.r[layer](torch.cat([input_, hidden[layer]], 1)))
                z = torch.sigmoid(self.z[layer](torch.cat([input_, hidden[layer]], 1)))
                h = torch.tanh(self.h[layer](torch.cat([input_, r*hidden[layer]], 1)))
                temp_hidden.append((1 - z) * hidden[layer] + z * h)
                # Apply dropout on this layer, but not for the recurrent units
                input_ = self.dropout(temp_hidden[-1])    
            # Store the output of the time step
            hidden = torch.stack(temp_hidden)
            logits[timestep] = self.out_layer(input_)
        return logits, hidden

    def generate(self, input1, hidden, generated_seq_len):
        """
        Generate a sample sequence from the GRU.

        This is similar to the forward method but instead of having ground
        truth input for each time step, you are now required to sample the token
        with maximum probability at each time step and feed it as input at the
        next time step.

        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """

        if input1.is_cuda:
            device = input1.get_device()
        else:
            device = torch.device("cpu")

        # Create a tensor to store outputs during the Forward
        logits = torch.zeros(generated_seq_len, self.batch_size).to(device)

        # print("**", input_.shape)
        # For each time step
        for timestep in range(generated_seq_len):
            input1 = self.word_embeddings(input1)

            # Apply dropout on the embedding result
            # For each layer
            for layer in range(self.num_layers):
                # Calculate the hidden states
                # And apply the activation function tanh on it
                r = torch.sigmoid(self.r[layer](torch.cat([input1, hidden[layer]], 1)))
                z = torch.sigmoid(self.z[layer](torch.cat([input1, hidden[layer]], 1)))
                h = torch.tanh(self.h[layer](torch.cat([input1, r*hidden[layer]], 1)))
                hidden[layer] = (1 - z) * hidden[layer] + z * h
                # # Apply dropout on this layer, but not for the recurrent units
                input1 = self.dropout(hidden[layer])

            input1 =  torch.argmax(torch.softmax(self.out_layer(input1),1), dim=1)            
            logits[timestep] = input1
        
        return logits


# Problem 2
##############################################################################
#
# Code for the Transformer models
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and
applying it to sequential language modelling. We use a binary "mask" to specify
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections,
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks,
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of input and output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units
        self.n_heads = n_heads
        # Create the layers below. self.linears should contain 3 linear
        # layers that compute the projection from n_units => n_heads x d_k
        # (one for each of query, key and value) plus an additional final layer
        # (4 in total)
        # Note: that parameters are initialized with Glorot initialization in
        # the make_model function below (so you don't need to implement this
        # yourself).

        # Note: the only Pytorch modules you are allowed to use are nn.Linear
        # and nn.Dropout. You can also use softmax, masked_fill and the "clones"
        # function we provide.
        # Size for last layer?? - NOT SURE

        self.linears = clones(nn.Linear(self.n_units, self.n_units), 4)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        # Implement scaled dot product attention
        # The query, key, and value inputs will be of size
        # batch_size x n_heads x seq_len x d_k
        # (If making a single call to attention in your forward method)
        # and mask (if not None) will be of size
        # batch_size x 1 x seq_len x seq_len

        # As described in the .tex, apply input masking to the softmax
        # generating the "attention values" (i.e. A_i in the .tex)

        # Also apply dropout to the attention values.
        # This method needs to compare query and keys first, then mask positions
        # if a mask is provided, normalize the scores, apply dropout and then
        # retrieve values, in this particular order.
        # When applying the mask, use values -1e9 for the masked positions.
        # The method returns the result of the attention operation as well as
        # the normalized scores after dropout.

        scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(self.d_k)
        if mask is not None:
            if len(mask.size()) == 3 and len(query.size()) == 4:
                mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        norm_scores = torch.softmax(scores, -1)
        if dropout is not None:
            norm_scores =  self.dropout(norm_scores)# Tensor of shape batch_size x n_heads x seq_len x seq_len
        output = torch.matmul(norm_scores,value)# Tensor of shape batch_size x n_heads x seq_len x d_k

        return output, norm_scores


    def forward(self, query, key, value, mask=None):
        # Implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # This method should call the attention method above
        if mask is not None:
            # Same mask applied to all n_heads heads.
            mask = mask.unsqueeze(1)
        b_size = query.size(0)
        # 1) Do all the linear projections in batch from n_units => n_heads x d_k
        q = self.linears[0](query).view(b_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k = self.linears[1](key).view(b_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v = self.linears[2](value).view(b_size, -1, self.n_heads, self.d_k).transpose(1,2)

        # 2) Apply attention on all the projected vectors in batch.
        # The query, key, value inputs to the attention method will be of size
        # batch_size x n_heads x seq_len x d_k
        output, norm_scores = self.attention(q, k, v, mask) 
        # 3) "Concat" using a view and apply a final linear.
        output = output.transpose(1, 2).contiguous().view(b_size, -1, self.n_units)
        output = self.linears[3](output)
        return output







#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
