import argparse
import boto3
import os
import logging
import sys

import random

import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
import itertools

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    logger.info(f"features: {features.shape}")
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i]] = 1
    return features



def verify_and_drop(df, column, mapping_dict):
    for index, row in df.iterrows():
        if row[column] not in mapping_dict:
            df.drop(index, inplace=True)
    return df



# Define Dataset class
class ChordsDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataframe, mapping):
        'Initialization'
        # Mapping
        self.chord_to_int = mapping[0]
        self.genre_to_int = mapping[1]
        self.key_to_int = mapping[2]
        self.artist_to_int = mapping[3]
        self.int_to_chord = mapping[4]
    
        df = dataframe
        
        # Apply verification and drop for each column in the dataframe
        df = verify_and_drop(df, "Chords", self.chord_to_int)
        df = verify_and_drop(df, "Genre", self.genre_to_int)
        df = verify_and_drop(df, "Key", self.key_to_int)
        df = verify_and_drop(df, "Artist", self.artist_to_int)
        
        # List of chords
        all_chords = df.Chords.to_list()[:1000]
        all_genres = df.Genre.to_list()[:1000]
        all_keys = df.Key.to_list()[:1000]
        all_artists = df.Artist.to_list()[:1000]

        # Convert chords, genres, keys, and artists to integers
        chords_int = [self.chord_to_int[chord] for chord in all_chords if chord in self.chord_to_int.keys()]  
        genres_int = [self.genre_to_int[genre] for genre in all_genres if genre in self.genre_to_int.keys()]
        keys_int = [self.key_to_int[key] for key in all_keys if key in self.key_to_int.keys()]
        artists_int = [self.artist_to_int[artist] for artist in all_artists if artist in self.artist_to_int.keys()]

        # Dataset of (chord, genre, key, artist, next chord) tuples
        all_chords_tuples = list(zip(chords_int, genres_int, keys_int, artists_int, chords_int[1:]))

        batch_size = len(all_chords) - 1
        seq_len = 1

        # input sequence and output sequence
        current_chords, current_genres, current_keys, current_artists, next_chords = zip(*all_chords_tuples)

        input_chords_encoded = one_hot_encode(current_chords, len(self.chord_to_int), seq_len, batch_size)
        input_genres_encoded = one_hot_encode(current_genres, len(self.genre_to_int), seq_len, batch_size)
        input_keys_encoded = one_hot_encode(current_keys, len(self.key_to_int), seq_len, batch_size)
        input_artists_encoded = one_hot_encode(current_artists, len(self.artist_to_int), seq_len, batch_size)

        self.dict_size = len(self.chord_to_int) + len(self.genre_to_int) + len(self.key_to_int) + len(self.artist_to_int)

        input_seq = np.concatenate((input_chords_encoded, input_genres_encoded, input_keys_encoded, input_artists_encoded), axis=2)

        self.input_seq = torch.Tensor(input_seq)
        self.target_seq = torch.Tensor(next_chords)
        
    
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.target_seq)

  def __getitem__(self, index):
        return self.input_seq[index], self.target_seq[index]
    
def prepare_mapping(dataframe):
        df = dataframe
        # List of chords
        all_chords = df.Chords.to_list()[:1000]
        all_genres = df.Genre.to_list()[:1000]
        all_keys = df.Key.to_list()[:1000]
        all_artists = df.Artist.to_list()[:1000]

        # Get unique chords
        unique_chords = list(set(all_chords))
        unique_genres = list(set(all_genres))
        unique_keys = list(set(all_keys))
        unique_artists = list(set(all_artists))

        # Genero como feature, Key como feature e Artist como feature
        chord_to_int = {chord:i for i, chord in enumerate(unique_chords)}
        genre_to_int = {genre: i for i, genre in enumerate(unique_genres)}
        key_to_int = {key: i for i, key in enumerate(unique_keys)}
        artist_to_int = {artist: i for i, artist in enumerate(unique_artists)}

        # Creating dictionaries that map integers to chords, genres, keys, and artists
        int_to_chord = dict(enumerate(unique_chords))
        
        return [chord_to_int, genre_to_int, key_to_int, artist_to_int, int_to_chord]
    
def create_data_loaders(args):
    
    # Connect to S3
    s3 = boto3.client('s3')
    Bucket = 'project-soungprogress-database'
    key_import_train = 'dataset/train/' + 'train.csv'
    key_import_test = 'dataset/test/' + 'test.csv'
    key_import_validation = 'dataset/validation/' + 'validation.csv'
    
    # Import train-validation dataset
    obj = s3.get_object(Bucket=Bucket, Key=key_import_train)
    df = pd.read_csv(obj['Body'])
    # mapping antes do treino para manter mesmas features
    maps = prepare_mapping(dataframe = df)
    train_dataset = ChordsDataset(dataframe=df, mapping=maps)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    
    # Import test dataset
    obj = s3.get_object(Bucket=Bucket, Key=key_import_test)
    df = pd.read_csv(obj['Body'])
    test_dataset = ChordsDataset(dataframe=df, mapping=maps)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=True)
    
    return train_loader, test_loader, train_dataset


# Define the model
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


    

def train(model, train_loader, criterion, optimizer, epoch, device):
    for e in range(epochs):
        for batch_idx, (features, target) in enumerate(train_loader):
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            output, hidden = model(features)
            output = output.to(device)
            loss = criterion(output, target.type(torch.LongTensor)) # Loss criterion
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
            if batch_idx % 100 == 0:
                logger.info(f"Train Epoch: {epoch} [({100.0 * batch_idx / len(train_loader)}%)]")

            
            
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for features, target in test_loader:
            output = model(features)
            test_loss += criterion(output, target.type(torch.LongTensor))  # sum up batch loss
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds==labels.data).item()
    avg_acc = running_corrects / len(test_loader.dataset)
    avg_loss = test_loss / len(test_loader.dataset)
    logger.info(f"Test set: Average loss: {avg_loss}, Average accuracy: {100*avg_acc}%")
    
    
def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

    
    

def main(args):

    # Criar os dataloaders
    train_loader, test_loader, dataset = create_data_loaders(args)
    
    # Config the device
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    # Instantiate the model with hyperparameters
    model = Model(input_size=dataset.dict_size, output_size=dataset.dict_size, hidden_dim=128, n_layers=1)
    model = model.to(device)
    
    # Define Loss and Optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    train(model, train_loader, args.epochs, loss_criterion, optimizer, epoch, device)
    
    # Test model
    test(model, test_loader, loss_criterion, device)
    
    # Save the trained model
    save_model(model, model_dir = 's3://project-soungprogress-database/model/')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default:2)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"],
        help="training data path in S3"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="location to save the model to"
    )
    
    
    args = parser.parse_args()
    
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Test Batch Size: {args.test_batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    
    main(args)