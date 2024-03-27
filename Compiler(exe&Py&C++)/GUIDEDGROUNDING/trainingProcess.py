from modelPool import ModelWrapper
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Function
import matplotlib.pyplot as plt
from extractTrainingDataFromJsonFile import *
from embeddingTrainingData import *
import numpy as np


# for training
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import torch.optim as optim
import torch.nn.functional as F

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

true = True
false = False
# for testing

# for collaboration

# Hyperparameters
CODE_EMBEDDING_SIZE = 55 - 3 * 8
INPUT_FLAG_SIZE = 1
# CODE_EMBEDDING_SIZE = int(
#     extended_code_embedding(
#         extended_grounding_state_list[0].extendedCodeTable.extendedCodeList[0]
#     ).__len__()
# )
code_segment_length = 20
input_size = code_segment_length * (
    CODE_EMBEDDING_SIZE
) + INPUT_FLAG_SIZE  # Assuming each code embedding has a size of YOUR_EMBEDDING_SIZE and codeSegmentLength=5
hidden_size = int(input_size*2)
output_size = int(20)
num_layers = 2
num_epochs = 30
batch_size = 16
learning_rate = 0.0001

pad_token = -11
heads = 20

code_segment_embedding_data = []  # store the codeseg and corresponding embedding

# Define the device here for global accessibility
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(MultiHeadAttention, self).__init__()
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads

#         assert (
#             self.head_dim * heads == embed_size
#         ), "Embedding size needs to be divisible by heads"

#         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

#     def forward(self, values, keys, query, mask):
#         N = query.shape[0]
#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

#         # Split the embedding into self.heads different pieces
#         values = values.reshape(N, value_len, self.heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
#         queries = query.reshape(N, query_len, self.heads, self.head_dim)

#         values = self.values(values)
#         keys = self.keys(keys)
#         queries = self.queries(queries)

#         # Scaled dot-product attention
#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (
#             self.embed_size ** (1 / 2)
#         )
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))

#         attention = torch.nn.functional.softmax(energy, dim=3)

#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
#             N, query_len, self.heads * self.head_dim
#         )

#         out = self.fc_out(out)
#         return out


def object_to_dict(obj):
    # If obj is an object, convert it to a dictionary
    if hasattr(obj, "__dict__"):
        obj_dict = {}
        # For each attribute, recursively convert it to a dictionary if it's an object
        for key, value in obj.__dict__.items():
            obj_dict[object_to_dict(key)] = object_to_dict(value)
        return obj_dict
    # If obj is a list, recursively convert its items
    elif isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    # If obj is a set, convert it to a list
    elif isinstance(obj, set):
        return object_to_dict(list(obj))
    # If obj is a PyTorch Tensor, convert it to a list
    elif isinstance(obj, torch.Tensor):
        return object_to_dict(obj.tolist())
    # Otherwise, return the object as it is
    else:
        return obj


class CodeSegmentDataset(Dataset):
    def __init__(
        self,
        extended_grounding_state_list: list[ExtendedGroundingState],
        code_segment_length=20,
        label_length=output_size,
        predict_goal="policies",
    ):
        self.data: list[ExtendedGroundingState] = extended_grounding_state_list
        self.code_segment_length = code_segment_length
        self.label_length = label_length
        self.predict_goal = predict_goal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        extended_grounding_state = self.data[idx]
        extended_code_table = extended_grounding_state.extendedCodeTable
        extended_code_embeddings = [
            torch.tensor(extended_code_embedding(extended_code) +
                         [pad_token])  # original to be -12
            for extended_code in extended_code_table.extendedCodeList[::]
        ]
        extended_code_embeddings[-1][-1] = pad_token  # -13

        # Make sure every input is of fixed length by padding or truncating
        extended_code_segments = []
        if len(extended_code_embeddings) <= self.code_segment_length:
            segment_tensor = extended_code_embeddings
            while (
                len(segment_tensor) < self.code_segment_length
            ):  # Just to ensure that the last segment has the required length, though it might not be necessary
                segment_tensor.append(
                    [pad_token] * segment_tensor[0].__len__())

            extended_code_segments = torch.cat(
                [torch.tensor(sublist) for sublist in segment_tensor], dim=0
            )
            extended_code_segments = torch.unsqueeze(
                extended_code_segments, dim=0)
        else:
            for i in range(
                0,
                len(extended_code_embeddings) - self.code_segment_length + 1,
                self.code_segment_length,
            ):
                segment_tensor = extended_code_embeddings[
                    i: min(i + self.code_segment_length,
                           len(extended_code_embeddings))
                ]
                while (
                    len(segment_tensor[-1]) < self.code_segment_length
                ):  # Just to ensure that the last segment has the required length, though it might not be necessary
                    segment_tensor[-1].extend(
                        [pad_token] *
                        (self.code_segment_length - len(segment_tensor))
                    )
                    # segment_tensor.append(torch.zeros_like(segment_tensor[0]))
                segment_tensor = torch.cat(
                    [sublist.clone().detach() for sublist in segment_tensor], dim=0
                )
                extended_code_segments.append(segment_tensor)
            extended_code_segments = torch.stack(extended_code_segments, dim=0)

        # Convert list of tensors to a single tensor with an extra sequence length dimension
        # extended_code_embeddings_tensor = torch.stack(extended_code_embeddings)
        label = None
        if self.predict_goal == "policies":
            label = output_embedding(extended_grounding_state)
            while label.__len__() < self.label_length:
                label.extend([0] * (self.label_length - label.__len__()))
            label = torch.tensor(label[::], dtype=torch.float32)

        # data_dict = {
        #     "extended_grounding_state": object_to_dict(extended_grounding_state),
        #     "extended_code_embeddings": object_to_dict(extended_code_embeddings),
        #     "extended_code_segments": extended_code_segments.tolist().__str__(),
        #     "labels": label.__str__(),
        # }
        # code_segment_embedding_data.append(data_dict)
        step_flagbit = torch.tensor(
            [extended_grounding_state.step]).unsqueeze(1)
        extended_code_segments = torch.cat(
            (step_flagbit, extended_code_segments), dim=1)

        if self.predict_goal == "policies":
            pass
        elif self.predict_goal == "advance":
            position_flagbit = torch.tensor(
                [extended_grounding_state.position]).unsqueeze(1)
            # append position to the first bit of the input data
            extended_code_segments = torch.cat(
                (position_flagbit, extended_code_segments), dim=1)  # now the input should be [pos, step, code embedding...]
            # modify label to be [advance==false, advance == true]
            if extended_grounding_state.advance == 0:
                label = torch.tensor([1, 0])
            else:
                label = torch.tensor([0, 1])
        elif self.predict_goal == "rpsign":
            position_flagbit = torch.tensor(
                [extended_grounding_state.position]).unsqueeze(1)
            # append position to the first bit of the input data
            extended_code_segments = torch.cat(
                (position_flagbit, extended_code_segments), dim=1)  # now the input should be [pos, step, code embedding...]
            # modify label to be [rpsign==false(negative), rpsign == true(positive)]
            if extended_grounding_state.rpsign == 0:
                label = torch.tensor([1, 0])
            else:
                label = torch.tensor([0, 1])

        return extended_code_segments, label

    def _one_hot_encode(self, position, size):
        # Helper function to convert position into one-hot vector
        one_hot = torch.zeros(size)
        one_hot[position] = 1
        return one_hot


def collate_fn(batch):
    """
    Custom function to collate data points into batches.
    """
    data, labels = zip(*batch)
    return torch.stack(data), torch.stack(labels)


def get_dataloader(
    extended_grounding_state_list: list[ExtendedGroundingState],
    code_segment_length=5,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    evaluate=False,
    predict_goal="policies",
    output_size=output_size
):
    """
    Custom dataloader to handle sequence data.
    """
    dataset = CodeSegmentDataset(
        extended_grounding_state_list, code_segment_length, predict_goal=predict_goal, label_length=output_size)
    sampler = None
    if shuffle == False and evaluate == False:
        if predict_goal == "policies":
            pass
        elif predict_goal == "advance":
            pass
        elif predict_goal == "rpsign":
            # oversample the negative ones
            # minority_count = sum(
            #     label.tolist() == [1, 0] for _, label in dataset)
            minority_count = 0
            minority_count = 0
            for idx, (_, label) in enumerate(dataset):
                # Assuming label is a tensor and you're interested in the first element
                if label.ndim > 0 and label.shape[0] > 0 and label[0] == 1:
                    minority_count += 1
                if idx >= dataset.__len__()-1:  # strange bugs
                    break

            total_count = len(dataset)
            desired_minority_prob = 0.33
            W_majority = 1
            W_minority = ((1 - desired_minority_prob) / desired_minority_prob) * \
                ((total_count - minority_count) / minority_count)

            # Assign weights
            # weights = torch.tensor(
            #     [W_minority if label[0] == 1 else W_majority for _, label in dataset])
            weights = []
            for idx, (_, label) in enumerate(dataset):
                # Check the condition for the minority class
                if label.ndim > 0 and label.shape[0] > 0 and label[0] == 1:
                    # Append the weight for the minority class
                    weights.append(W_minority)
                else:
                    # Append the weight for the majority class
                    weights.append(W_majority)
                if idx >= dataset.__len__()-1:  # strange bugs
                    break
            weights = torch.tensor(weights)
            # Create the sampler and DataLoader
            sampler = WeightedRandomSampler(
                weights, len(weights), replacement=True)
    if shuffle == True:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            sampler=sampler,  # Deliberately generating error alerts
        )
    else:
        if evaluate == True:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            sampler=sampler
        )
    return dataloader


class CodeLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers=4, code_segment_length=5
    ):
        super(CodeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.code_segment_length = code_segment_length

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, bidirectional=True)

        # self.attention = MultiHeadAttention(hidden_size, heads=heads)

        # Define the output layer
        self.linear = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        # x is already in the shape (batch_size, codeSegmentLength, input_size)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(
            0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))
        # out = self.attention(out, out, out, None)
        # # Use only the output from the last time step
        # out = self.linear(out[:, -1, :])
        out_forward = out[:, -1, :self.hidden_size]
        out_backward = out[:, 0, self.hidden_size:]
        out_reduced = torch.cat((out_forward, out_backward), 1)

        # Decode the concatenated final states
        out = self.linear(out_reduced)

        return out


################
# run
################
class CustomLossFunction(Function):
    @staticmethod
    def forward(ctx, predictions, labels):
        _, predicted_indices = torch.max(predictions, dim=1)
        _, label_indices = torch.max(labels, dim=1)

        cost = torch.where(
            predicted_indices != label_indices,
            torch.tensor(1.0, device=predictions.device),
            torch.tensor(0.0, device=predictions.device),
        )

        # Save tensors for backward pass
        ctx.save_for_backward(
            predicted_indices, label_indices, predictions, labels)

        return cost.mean()

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve tensors from the forward pass
        predicted_indices, label_indices, predictions, labels = ctx.saved_tensors

        # Initialize the gradient tensor with zeros
        grad_predictions = torch.zeros_like(predictions)

        differences = predictions - labels
        # Update the gradient tensor at the predicted indices positions
        for i, (pred_idx, label_idx) in enumerate(
            zip(predicted_indices, label_indices)
        ):
            grad_predictions[i] = 10 * differences[i] * \
                torch.exp(-differences[i])
            grad_predictions[i, label_indices] *= pred_idx != label_idx

        return grad_predictions, None


def evaluate(model, test_dataloader_, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for sequences, labels in test_dataloader_:
            sequences = sequences.to(device)
            mask = (sequences != pad_token).float().to(device)
            sequences = sequences * mask

            labels = labels.to(device)
            label_mask = (labels != pad_token).float().to(device)
            labels = labels * label_mask

            outputs = model(sequences)
            outputs = label_mask * outputs

            loss = criterion(outputs, labels)
            loss = (loss * mask).sum() / mask.sum()
            total_loss += loss.item()

            # Convert the outputs to probabilities
            probs = torch.softmax(outputs, dim=1)
            predictions = (probs).float()

            # Convert probabilities to binary predictions
            _, predicted = torch.max(probs.data, 1)
            _, labels_indices = torch.max(labels, 1)

            total_correct += (predicted == labels_indices).sum().item()

    avg_loss = total_loss / len(test_dataloader_)
    accuracy = total_correct / (
        len(test_dataloader_) * test_dataloader_.batch_size
    )

    return avg_loss, accuracy


#######
# training and testing
#######
def trainLSTM(trainingDataFilePath: str = None, trainingDataList=None, model=None,
              input_size=input_size,
              hidden_size=hidden_size,
              output_size=output_size,
              num_layers=num_layers,
              predict_goal="policies"):
    torch.cuda.empty_cache()
    grounding_state_list = []
    if trainingDataFilePath != None:
        grounding_state_list += initialize_from_file(trainingDataFilePath)
    if trainingDataList != None:
        grounding_state_list += trainingDataList
    extended_grounding_state_list = [
        ExtendedGroundingState(grounding_state)
        for grounding_state in grounding_state_list
    ]
    # Calculate lengths of splits
    test_extended_grounding_state_list_length = int(
        len(extended_grounding_state_list) * 0.10
    )
    train_extended_grounding_state_list_length: int = (
        len(extended_grounding_state_list) -
        test_extended_grounding_state_list_length
    )
    # Randomly split dataset into training set and test set
    train_dataset, test_dataset = random_split(
        extended_grounding_state_list,
        [
            train_extended_grounding_state_list_length,
            test_extended_grounding_state_list_length,
        ],
    )

    ######
    # training part
    ######
    # Load data
    train_dataloader = None
    if predict_goal == "policies":
        train_dataloader = get_dataloader(
            train_dataset,
            code_segment_length=code_segment_length,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            predict_goal=predict_goal,
            output_size=output_size,
            evaluate=False,
        )
        test_dataloader = get_dataloader(
            test_dataset,
            code_segment_length=code_segment_length,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            predict_goal=predict_goal,
            output_size=output_size,
            evaluate=True,
        )
    elif predict_goal == "advance":
        train_dataloader = get_dataloader(
            train_dataset,
            code_segment_length=code_segment_length,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            predict_goal=predict_goal,
            output_size=output_size,
            evaluate=False,
        )
        test_dataloader = get_dataloader(
            test_dataset,
            code_segment_length=code_segment_length,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            predict_goal=predict_goal,
            output_size=output_size,
            evaluate=True,
        )
    elif predict_goal == "rpsign":
        train_dataloader = get_dataloader(
            train_dataset,
            code_segment_length=code_segment_length,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            predict_goal=predict_goal,
            output_size=output_size,
            evaluate=False,
        )
        test_dataloader = get_dataloader(
            test_dataset,
            code_segment_length=code_segment_length,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            predict_goal=predict_goal,
            output_size=output_size,
            evaluate=True,
        )
    else:
        assert (False)

    # Instantiate model, loss and optimizer
    if model == None:
        model = CodeLSTM(input_size, hidden_size,
                         output_size, num_layers).to(device)
    criterion = (
        # nn.BCEWithLogitsLoss()
        nn.CrossEntropyLoss()
    )  # This combines a sigmoid layer and the BCE loss, good for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for param in model.parameters():
        assert (
            param.requires_grad == True
        ), "param.requires_grad == false in model, exit"

    # Train the model
    model.train()
    loss_function = nn.CrossEntropyLoss()  # CustomLossFunction.apply
    # This dictionary will store models based on accuracy
    saved_models = {}
    # The best model details
    best_model = {"model_state": None, "accuracy": 0.0}
    # Lists to store the loss and accuracy values for plotting
    loss_values = []
    accuracy_values = []
    dataLoader_data = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # for calculating average loss for the epoch
        epoch_correct = 0  # for calculating average accuracy for the epoch
        total_samples = 0  # total samples in the epoch
        for i, (sequences, labels) in enumerate(train_dataloader):
            data_dict = {
                "epoch": epoch,
                "index": i,
                "sequences": sequences.cpu().numpy().tolist(),
                "labels": labels.cpu().numpy().tolist(),
            }
            dataLoader_data.append(data_dict)

            sequences = sequences.to(device)
            labels = labels.to(device)
            # mask
            mask = (sequences != pad_token).float().to(device)
            sequences = sequences * mask

            label_mask = (labels != pad_token).float().to(device)
            labels = labels * label_mask

            # Forward pass
            outputs = model(sequences)
            outputs = label_mask * outputs

            loss = loss_function(outputs, labels)
            loss = (loss * mask).sum() / mask.sum()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            _, labels_indices = torch.max(labels, 1)

            total = labels.size(1)
            correct = (predicted == labels_indices).sum().item()
            accuracy = correct / predicted.size()[0]

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                loss_values.append(loss.item())
                accuracy_values.append(accuracy)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
                )
            epoch_loss += loss.item()
            epoch_correct += correct
            total_samples += sequences.size(0)
        scheduler.step()
        avg_loss = epoch_loss / len(train_dataloader)
        avg_accuracy = epoch_correct / total_samples
    #     if epoch % 1 == 0:
    #         saved_models[avg_accuracy] = {
    #             "model_state": model.state_dict(),
    #             "loss": avg_loss,
    #             "accuracy": avg_accuracy,
    #         }
    #         if avg_accuracy > best_model["accuracy"]:
    #             best_model["model_state"] = model.state_dict()
    #             best_model["accuracy"] = avg_accuracy
    # # Set the best model as the final model
    #     model.load_state_dict(best_model["model_state"])
    # with open("dataloader_data.json", "w") as f:
    #     json.dump(dataLoader_data, f)

    ######
    # test
    ######
    test_loss, test_accuracy = evaluate(model, test_dataloader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # draw curves
    # Plotting Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, color="red")
    plt.xlabel("Steps (in tens)")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.grid(True)
    plt.show(block=True)

    # Plotting Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_values, color="blue")
    plt.xlabel("Steps (in tens)")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy over Time")
    plt.grid(True)
    plt.show(block=True)

    return model, test_loss, test_accuracy


class LSTMGuidedReasoning(ModelWrapper):
    def __init__(self) -> None:
        self.model_policies = CodeLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            code_segment_length=code_segment_length,
        ).to(device)
        self.model_policies_accuracy = 0

        self.model_advance = CodeLSTM(
            input_size=input_size+1,
            hidden_size=2*(input_size+1),
            output_size=2,
            num_layers=num_layers,
            code_segment_length=code_segment_length,
        ).to(device)
        self.model_advance_accuracy = 0

        self.model_rpsign = CodeLSTM(
            input_size=input_size+1,
            hidden_size=2*(input_size+1),
            output_size=2,
            num_layers=num_layers,
            code_segment_length=code_segment_length,
        ).to(device)
        self.model_rpsign_accuracy = 0
        pass

    def train(
        self, trainingDataFilePath=None, trainingDataList: list = None
    ) -> tuple[str, str]:
        self.model_policies, self.model_policies_loss, self.model_policies_accuracy = trainLSTM(
            trainingDataFilePath=trainingDataFilePath,
            trainingDataList=trainingDataList,
            model=self.model_policies,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            predict_goal="policies",
        )

        self.model_advance, self.model_advance_loss, self.model_advance_accuracy = trainLSTM(
            trainingDataFilePath=trainingDataFilePath,
            trainingDataList=trainingDataList,
            model=self.model_advance,
            input_size=input_size + 1,
            hidden_size=2*(input_size + 1),
            output_size=2,
            num_layers=num_layers,
            predict_goal="advance",
        )

        self.model_rpsign, self.model_rpsign_loss, self.model_rpsign_accuracy = trainLSTM(
            trainingDataFilePath=trainingDataFilePath,
            trainingDataList=trainingDataList,
            model=self.model_rpsign,
            input_size=input_size + 1,
            hidden_size=2*(input_size + 1),
            output_size=2,
            num_layers=num_layers,
            predict_goal="rpsign",
        )

        return self.model_policies_loss, self.model_policies_accuracy, self.model_advance_loss, self.model_advance_accuracy, self.model_rpsign_loss, self.model_rpsign_accuracy

    def load_model(self, model_path: str):
        model_checkpoint = torch.load(model_path)
        self.model_policies.load_state_dict(
            model_checkpoint['model_policies_state_dict'])
        self.model_policies_accuracy = model_checkpoint['model_policies_accuracy']
        self.model_advance.load_state_dict(
            model_checkpoint['model_advance_state_dict'])
        self.model_advance_accuracy = model_checkpoint['model_advance_accuracy']
        self.model_rpsign.load_state_dict(
            model_checkpoint['model_rpsign_state_dict'])
        self.model_rpsign_accuracy = model_checkpoint['model_rpsign_accuracy']

    # def load_state_dict(self, state_dict):
    #     self.model.load_state_dict(state_dict)

    def predict(
        self, grounding_state_str: str, predicted_position_output_file
    ):
        self.json_result = {}
        self.predict_policies(grounding_state_str,
                              predicted_position_output_file)
        self.predict_advance(grounding_state_str,
                             predicted_position_output_file)
        self.predict_rpsign(grounding_state_str,
                            predicted_position_output_file)
        return self.json_result

    def predict_policies(
        self, grounding_state_str: str, predicted_position_output_file
    ):

        eval_extended_grounding_state = ExtendedGroundingState(
            GroundingState(**json.loads(grounding_state_str))
        )
        eval_dataset = [eval_extended_grounding_state]
        # predicted_position_output_file must be an opening file
        # with open('test_eval_data.json', 'r') as file:
        #     eval_dataset = [ExtendedGroundingState(GroundingState(**entry))
        #                     for entry in json.load(file)]

        self.model_policies.eval()  # Important! Set the model to evaluation mode
        eval_dataloader = get_dataloader(
            eval_dataset,
            code_segment_length=code_segment_length,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            evaluate=True,
            predict_goal="policies",
            output_size=output_size,
        )
        with torch.no_grad():
            for i, (sequences, labels) in enumerate(eval_dataloader):  # only loop once

                # sequences = sequences.to(device)
                sequences = sequences.to(device)
                mask = (sequences != pad_token).float().to(device)
                sequences = sequences * mask

                labels = labels.to(device)
                label_mask = (labels != pad_token).float().to(device)
                labels = labels * label_mask

                outputs = self.model_policies(sequences)
                outputs = label_mask * outputs

                # predict_outputs = outputs[0][predicted_result_index]
                softmax = nn.Softmax(dim=1)
                probs = softmax(outputs)

                # Convert probabilities to binary predictions
                _, predicted = torch.max(outputs.data, 1)
                _, labels_indices = torch.max(labels, 1)

                total = labels.size(1)
                correct = (predicted == labels_indices).sum().item()
                accuracy = correct / total

                _, predicted_indices = torch.max(probs.data, dim=1)
                predictions = predicted_indices.float()
                print(
                    f"predictions:{predictions.tolist()}\nprobs:{probs.tolist()}")
                break
            self.json_result["policies"] = probs[0].tolist()
            self.json_result["ac_policies"] = float(
                self.model_policies_accuracy)
        return

    def predict_advance(
        self, grounding_state_str: str, predicted_position_output_file
    ):

        eval_extended_grounding_state = ExtendedGroundingState(
            GroundingState(**json.loads(grounding_state_str))
        )

        ####### important####
        eval_extended_grounding_state.position = self.json_result["policies"].index(
            max(self.json_result["policies"]))
        eval_dataset = [eval_extended_grounding_state]
        # predicted_position_output_file must be an opening file
        # with open('test_eval_data.json', 'r') as file:
        #     eval_dataset = [ExtendedGroundingState(GroundingState(**entry))
        #                     for entry in json.load(file)]

        self.model_advance.eval()  # Important! Set the model to evaluation mode
        eval_dataloader = get_dataloader(
            eval_dataset,
            code_segment_length=code_segment_length,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            evaluate=True,
            predict_goal="advance",
            output_size=2,
        )
        with torch.no_grad():
            for i, (sequences, labels) in enumerate(eval_dataloader):  # only loop once

                # sequences = sequences.to(device)
                sequences = sequences.to(device)
                mask = (sequences != pad_token).float().to(device)
                sequences = sequences * mask

                labels = labels.to(device)
                label_mask = (labels != pad_token).float().to(device)
                labels = labels * label_mask

                outputs = self.model_advance(sequences)
                outputs = label_mask * outputs

                # predict_outputs = outputs[0][predicted_result_index]
                softmax = nn.Softmax(dim=1)
                probs = softmax(outputs)

                # Convert probabilities to binary predictions
                _, predicted = torch.max(outputs.data, 1)
                _, labels_indices = torch.max(labels, 1)

                total = labels.size(1)
                correct = (predicted == labels_indices).sum().item()
                accuracy = correct / total

                _, predicted_indices = torch.max(probs.data, dim=1)
                predictions = predicted_indices.float()
                print(
                    f"predictions:{predictions.tolist()}\nprobs:{probs.tolist()}")
                break

            self.json_result["advance"] = probs[0].tolist()[1]
            self.json_result["ac_advance"] = float(
                self.model_advance_accuracy)
        return

    def predict_rpsign(
        self, grounding_state_str: str, predicted_position_output_file
    ):

        eval_extended_grounding_state = ExtendedGroundingState(
            GroundingState(**json.loads(grounding_state_str))
        )
        ####### important####
        eval_extended_grounding_state.position = self.json_result["policies"].index(
            max(self.json_result["policies"]))
        eval_dataset = [eval_extended_grounding_state]
        # predicted_position_output_file must be an opening file
        # with open('test_eval_data.json', 'r') as file:
        #     eval_dataset = [ExtendedGroundingState(GroundingState(**entry))
        #                     for entry in json.load(file)]

        self.model_rpsign.eval()  # Important! Set the model to evaluation mode
        eval_dataloader = get_dataloader(
            eval_dataset,
            code_segment_length=code_segment_length,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            evaluate=True,
            predict_goal="rpsign",
            output_size=2,

        )
        with torch.no_grad():
            for i, (sequences, labels) in enumerate(eval_dataloader):  # only loop once

                # sequences = sequences.to(device)
                sequences = sequences.to(device)
                mask = (sequences != pad_token).float().to(device)
                sequences = sequences * mask

                labels = labels.to(device)
                label_mask = (labels != pad_token).float().to(device)
                labels = labels * label_mask

                outputs = self.model_rpsign(sequences)
                outputs = label_mask * outputs

                # predict_outputs = outputs[0][predicted_result_index]
                softmax = nn.Softmax(dim=1)
                probs = softmax(outputs)

                # Convert probabilities to binary predictions
                _, predicted = torch.max(outputs.data, 1)
                _, labels_indices = torch.max(labels, 1)

                total = labels.size(1)
                correct = (predicted == labels_indices).sum().item()
                accuracy = correct / total

                _, predicted_indices = torch.max(probs.data, dim=1)
                predictions = predicted_indices.float()
                print(
                    f"predictions:{predictions.tolist()}\nprobs:{probs.tolist()}")
                break

            self.json_result["rpsign"] = probs[0].tolist()[1]
            self.json_result["ac_rpsign"] = float(
                self.model_rpsign_accuracy)
        return

    def get_state_dict(self):
        return self.model_policies.state_dict()

    def load_state_dict(self, state_dict):
        pass

    def save_model(self, model_path: str):

        # Extract the directory path from the model_path
        model_directory = os.path.dirname(model_path)
        # Check if the directory exists
        if model_directory != '' and not os.path.exists(model_directory):
            # Create the directory if it does not exist
            os.makedirs(model_directory)

        model_checkpoint = {
            'model_policies_state_dict': self.model_policies.state_dict(),
            'model_policies_accuracy': self.model_policies_accuracy,
            'model_advance_state_dict': self.model_advance.state_dict(),
            'model_advance_accuracy': self.model_advance_accuracy,
            'model_rpsign_state_dict': self.model_rpsign.state_dict(),
            'model_rpsign_accuracy': self.model_rpsign_accuracy,
        }
        torch.save(model_checkpoint, model_path)


if __name__ == "__main__":

    test_grounding_state_json_entry = {
        "BDDB_value": 1.7371074393165296,
        "codeTable": [
            {
                "grounded": false,
                "operand1": {
                    "argName": "8",
                    "argType": "number",
                    "changeable": 0,
                    "className": "",
                    "isClass": 0
                },
                "operand2": {
                    "argName": "x",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                },
                "operator": {
                    "argName": "*",
                    "argType": "other"
                },
                "result": {
                    "argName": "1944.3",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                }
            },
            {
                "grounded": false,
                "operand1": {
                    "argName": "x",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                },
                "operand2": {
                    "argName": "2",
                    "argType": "number",
                    "changeable": 0,
                    "className": "",
                    "isClass": 0
                },
                "operator": {
                    "argName": "^",
                    "argType": "other"
                },
                "result": {
                    "argName": "1944.4",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                }
            },
            {
                "grounded": false,
                "operand1": {
                    "argName": "-1",
                    "argType": "number",
                    "changeable": 0,
                    "className": "",
                    "isClass": 0
                },
                "operand2": {
                    "argName": "1944.3",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                },
                "operator": {
                    "argName": "*",
                    "argType": "other"
                },
                "result": {
                    "argName": "1944.5",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                }
            },
            {
                "grounded": false,
                "operand1": {
                    "argName": "1944.4",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                },
                "operand2": {
                    "argName": "1944.5",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                },
                "operator": {
                    "argName": "+",
                    "argType": "other"
                },
                "result": {
                    "argName": "1944.6",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                }
            },
            {
                "grounded": false,
                "operand1": {
                    "argName": "1944.6",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                },
                "operand2": {
                    "argName": "-9",
                    "argType": "number",
                    "changeable": 0,
                    "className": "",
                    "isClass": 0
                },
                "operator": {
                    "argName": "+",
                    "argType": "other"
                },
                "result": {
                    "argName": "1945",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                }
            },
            {
                "grounded": false,
                "operand1": {
                    "argName": "1945",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                },
                "operand2": {
                    "argName": "0",
                    "argType": "number",
                    "changeable": 0,
                    "className": "",
                    "isClass": 0
                },
                "operator": {
                    "argName": "==",
                    "argType": "other"
                },
                "result": {
                    "argName": "1946",
                    "argType": "identifier",
                    "changeable": 1,
                    "className": "",
                    "isClass": 0
                }
            }
        ],
        "initialState": false,
        "position": 3,
        "relatedFilesAndClasses": [
            "quadratic#1"
        ],
        "succeed": true
    }

    print(f"loading tokenizer...")
    load_operand_tokenizer(
        "operand_tokenizer.json"
    )
    print(f"tokenizer loaded")
    predict_output_file = open("predict_result.txt", "w")
    model_handler = LSTMGuidedReasoning()
    model_handler.train("trainingData.json")
    model_handler.save_model("model_checkpoint.mdl")
    model_handler.load_model("model_checkpoint.mdl")
    model_handler.predict(json.dumps(
        test_grounding_state_json_entry), predict_output_file)

    save_operand_tokenizer("operand_tokenizer.json")
