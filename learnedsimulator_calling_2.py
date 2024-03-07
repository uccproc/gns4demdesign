import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, radius_graph, GENConv, DeepGCNLayer
from torch.nn import Linear, LayerNorm, ReLU
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime#yujiang, 22.01.06, added
import shutil
import glob
# import timing
import time
# from labml_nn.graphs.gat import GraphAttentionLayer

start_time_0 = time.clock()
my_data_path = r'F:\dataset_gns\17_granular_2D_50_friction_restitution_DoE_2nd'#yujiang, 22.01.06, added
my_model_path = r'E:/OneDrive - University College Cork/recap_2310_12/GNS_for_2D_Dambreak_frictionrestitution/models/17_granular_2D_50_friction_restitution_DoE_2nd/' #make sure / is included, yujiang, 22.01.06, added
my_rollouts_path = r'E:/OneDrive - University College Cork/recap_2310_12/GNS_for_2D_Dambreak_frictionrestitution/rollouts/17_granular_2D_50_friction_restitution_DoE_2nd/'#yujiang, 22.01.06, added
topic = 'case8_mp8_r005_input5_'#yujiang, 22.01.15, added
my_rollouts_path_topic = os.path.join(my_rollouts_path, topic+'0') #yujiang, change '1' to 'any number' corresponding to model directory postfix
os.makedirs(my_model_path, exist_ok=True) #yujiang, 22.01.06, previously 'train_log'
os.makedirs(my_rollouts_path_topic, exist_ok=True)

default_connectivity_radius = 0.005 #yujiang, added
INPUT_SEQUENCE_LENGTH = 5
kinematic_particle_id = 3 #yujiang, this is the particle id that stands for fixed particles in the field
batch_size = 1
noise_std = 1e-4 #yujiang, 22.01.06, originally 6.7e-4
training_steps = int(1e6) #yujiang, 22.01.06, originally 2e7
log_steps = 5
eval_steps = 500
save_steps = 5000
diameter = 0.00042 #yujiang, the diameter of particles in DEM, corresponding to line 333
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #'cuda', or 'cpu', or None, #yujiang
device = torch.device('cuda:0')
sample_n_steps = 1 # yujiang, sampling position every n steps from tfrecord

is_train = False #yujiang, 22.01.06, added
model_path = r'E:\OneDrive - University College Cork\recap_2310_12\GNS_for_2D_Dambreak_frictionrestitution\models\17_granular_2D_50_friction_restitution_DoE_2nd\case8_mp8_r005_input5_0\case8_mp8_r005_input5_74000checkpoint_evalbest.pth.tar' #yujiang
# model_path = None # 'model425000.pth', or None

with open(os.path.join(my_data_path, 'metadata.json'), 'rt') as f:#yujiang, 22.01.06
    metadata = json.loads(f.read())
num_steps = metadata['sequence_length']//sample_n_steps - INPUT_SEQUENCE_LENGTH # yujiang, 2022.02.03 changed
normalization_stats = {
    'acceleration': {
        'mean':torch.FloatTensor(metadata['acc_mean']),
        'std':torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 + noise_std**2),
    },
    'velocity': {
        'mean':torch.FloatTensor(metadata['vel_mean']),
        'std':torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 + noise_std**2),
    },
}

def build_mlp(
    input_size,
    layer_sizes,
    output_size=None,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
):
    sizes = [input_size] + layer_sizes
    if output_size:
        sizes.append(output_size)

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]

def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = time_diff(position_sequence)
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5)

    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    return position_sequence_noise

def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def save_checkpoint(state, eval_is_best, is_best, best_eval_path, best_train_path, LOG_DIR):
    if eval_is_best:
        torch.save(state, best_eval_path)
    if is_best:
        for infile in glob.glob(os.path.join(LOG_DIR, '*trainbest.pth.tar')):
            os.remove(infile)
        torch.save(state, best_train_path) #yujiang,  'shutil.copyfile(checkpoint_path, best_model_path)'

class Encoder(nn.Module):
    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Encoder, self).__init__()
        self.node_fn = nn.Sequential(*[build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out),
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out),
            nn.LayerNorm(edge_out)])

    def forward(self, x, edge_index, e_features): # global_features
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        return self.node_fn(x), self.edge_fn(e_features)

class InteractionNetwork(MessagePassing):
    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(InteractionNetwork, self).__init__(aggr='add')
        # self.gatlayer1 = GraphAttentionLayer(node_in+edge_out, node_out, 1)
        self.node_fn = nn.Sequential(*[build_mlp(node_in+edge_out, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out),
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(node_in+node_in+edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out),
            nn.LayerNorm(edge_out)])

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        #self.adj_mat = torch.eye(x.size(dim=0), dtype=torch.bool)
        #index = (edge_index[0, :], edge_index[1, :])
        #new_value = torch.ones(edge_index.size(dim=1), dtype=torch.bool)
        #self.adj_mat.index_put_(index, new_value)
        # self.adj_mat = torch.unsqueeze(self.adj_mat, dim=2)
        # self.adj_mat = self.adj_mat.clone().detach().requires_grad_(False).to(device)

        x_residual = x
        e_features_residual = e_features
        x, e_features = self.propagate(edge_index=edge_index, x=x, e_features=e_features)
        return x+x_residual, e_features+e_features_residual

    def message(self, edge_index, x_i, x_j, e_features):
        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_fn(e_features)
        return e_features

    def update(self, x_updated, x, e_features):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        #x_updated = self.gatlayer1(x_updated, self.adj_mat)
        return x_updated, e_features

class Processor(MessagePassing):
    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        edge_out,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Processor, self).__init__(aggr='add')#yujiang, changed from 'max'
        #following is the GCN/gcn type of Processor part, yujiang, 2022.02.03
        #original, message passing gnn
        self.gnn_stacks = nn.ModuleList([
            InteractionNetwork(
                node_in=node_in,
                node_out=node_out,
                edge_in=edge_in,
                edge_out=edge_out,
                mlp_num_layers=mlp_num_layers,
                mlp_hidden_dim=mlp_hidden_dim,
            ) for _ in range(num_message_passing_steps)])

    def forward(self, x, edge_index, e_features):
        #following is the GCN part of forward
        for gnn in self.gnn_stacks:
            x, e_features = gnn(x, edge_index, e_features)
        return x, e_features

class Decoder(nn.Module):
    def __init__(
        self,
        node_in,
        node_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out)

    def forward(self, x):
        # x: (E, node_in)
        return self.node_fn(x)#delete .to(device), 9/28/2022

class EncodeProcessDecode(nn.Module):
    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ):
        super(EncodeProcessDecode, self).__init__()
        self._device = device
        self._encoder = Encoder(
            node_in=node_in,
            node_out=latent_dim,
            edge_in=edge_in,
            edge_out=latent_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = Processor(
            node_in=latent_dim,
            node_out=latent_dim,
            edge_in=latent_dim,
            edge_out=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._decoder = Decoder(
            node_in=latent_dim,
            node_out=node_out,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        x, e_features = self._encoder(x, edge_index, e_features)
        x, e_features = self._processor(x, edge_index, e_features)
        x = self._decoder(x)
        return x

class Simulator(nn.Module):
    def __init__(
        self,
        particle_dimension,
        node_in,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
        connectivity_radius,
        boundaries,
        normalization_stats,
        num_particle_types,
        particle_type_embedding_size,
        input_paramDict,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    ):
        super(Simulator, self).__init__()
        self._device = device
        self._boundaries = boundaries
        self._connectivity_radius = connectivity_radius
        self._normalization_stats = normalization_stats
        self._num_particle_types = num_particle_types
        self._input_paramDict = input_paramDict  ## Added by Yu, 2023.09
        self._particle_type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size) # (9, 16)

        self._encode_process_decode = EncodeProcessDecode(
            node_in=node_in,
            node_out=particle_dimension,
            edge_in=edge_in,
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self):
        pass

    def _build_graph_from_raw(self, position_sequence, n_particles_per_example, particle_types, friction, restitution):
        n_total_points = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1] # (n_nodes, 2)
        velocity_sequence = time_diff(position_sequence)
        # senders and receivers are integers of shape (E,)
        radius = torch.tensor(self._connectivity_radius).to(self._device)
        senders, receivers = self._compute_connectivity(most_recent_position, n_particles_per_example, radius)

        node_features = []
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self._normalization_stats["velocity"]
        velocity_stats['mean'] = velocity_stats['mean'].clone().detach().to(self._device)
        velocity_stats['std'] = velocity_stats['std'].clone().detach().to(self._device)
        normalized_velocity_sequence = (velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
        flat_velocity_sequence = normalized_velocity_sequence.view(n_total_points, -1)
        node_features.append(flat_velocity_sequence)

        # Normalized clipped distances to lower and upper boundaries.
        # boundaries are an array of shape [num_dimensions, 2], where the second
        # axis, provides the lower/upper boundaries.
        boundaries = torch.tensor(self._boundaries, requires_grad=False).float().to(self._device)
        distance_to_lower_boundary = (most_recent_position - boundaries[:, 0][None])
        distance_to_upper_boundary = (boundaries[:, 1][None] - most_recent_position)
        distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
        normalized_clipped_distance_to_boundaries = torch.clamp(distance_to_boundaries / self._connectivity_radius, -1., 1.)
        node_features.append(normalized_clipped_distance_to_boundaries)

        if self._num_particle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)
#        print('friction is', friction)
#        print('particle types is', particle_types)
#        print('particle_type_embeddings is', particle_type_embeddings)
#         node_features.append(friction.view(friction.shape[0], 1))
#         node_features.append(restitution.view(restitution.shape[0], 1))
        friction_value = self._input_paramDict['friction']
        restitution_value = self._input_paramDict['restitution']
        # friction_value = self._input_paramDict[0]
        # restitution_value = self._input_paramDict[1]
        friction = torch.full((len(friction),), friction_value).to(device)
        restitution = torch.full((len(restitution),), restitution_value).to(device)
        node_features.append(friction.view(friction.shape[0], 1))
        node_features.append(restitution.view(restitution.shape[0], 1))
        # node_features.append(width.view(width.shape[0], 1))
        # node_features.append(ratio.view(ratio.shape[0], 1))

        # Collect edge features.
        edge_features = []

        # Relative displacement and distances normalized to radius
        # (E, 2)
        # normalized_relative_displacements = (
        #     torch.gather(most_recent_position, 0, senders) - torch.gather(most_recent_position, 0, receivers)
        # ) / self._connectivity_radius
        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)
    

        #following is the (xi-xj)/(||xi-xj||2), 1/(||xi-xj||2), and two others added to the edge embedding, 22.01.24, yujiang
        edge_features.append(torch.div(1.0, torch.norm(normalized_relative_displacements, p = 1, dim=-1, keepdim=True)+0.0001).to(self._device))
        edge_features.append(torch.div(normalized_relative_displacements, normalized_relative_distances+0.0001).to(self._device))
        edge_features.append(torch.div(1.0, normalized_relative_distances+0.0001).to(self._device))
        edge_features.append(torch.div(normalized_relative_displacements, torch.norm(normalized_relative_displacements, p = 3, dim=-1, keepdim=True)+0.0001).to(self._device))
        #print(senders, "senders")
        #print(most_recent_position, "most_recent_position")

        #following is the overlap added to the edge embedding
        displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :])
        overlap_distances = torch.norm(displacements, dim=-1, keepdim=True) - diameter#here diameter is set in the begining, it is the diameter of 1 particle in DEM dataset
        overlap_distances = overlap_distances.clone().detach().cpu() # now it's in cpu, so the for loop could be quick
        #for i in range(len(overlap_distances)): #yujiang, added, 22.1.14, for adding overlap to the edge embedding
        #    if overlap_distances[i] > 1e-4:
        #        overlap_distances[i] = 0
#        edge_features.append(overlap_distances/self._connectivity_radius.to(self._device)) #yujiang, comment this if not needed

        #following is the relative velocity, added to edge_features
        most_recent_velocity = velocity_sequence[:, -1]#[? 2]
        velocity_stats = self._normalization_stats["velocity"]
        normalized_most_recent_velocity = (
                most_recent_velocity - velocity_stats["mean"])/velocity_stats["std"]
        normalized_relative_velocity = normalized_most_recent_velocity[senders, :] - normalized_most_recent_velocity[receivers, :]
#        edge_features.append(normalized_relative_velocity.to(self._device)) #yujiang, comment this if don't want to add this to edge feature
#        print('node_features',node_features)
#        print('senders',senders)
#        print('edge_features',edge_features)
        return torch.cat(node_features, dim=-1), torch.stack([senders, receivers]), torch.cat(edge_features, dim=-1)

    def _compute_connectivity(self, node_features, n_particles_per_example, radius, add_self_edges=True):
        # handle batches. Default is 2 examples per batch

        # Specify examples id for particles/points
        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_example)])
        # batch_ids = torch.tensor(batch_ids).to(self._device)
        batch_ids = batch_ids.clone().detach().to(self._device)
        #print('batch_ids',batch_ids)
        # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        edge_index = radius_graph(node_features, r=radius, batch=batch_ids, loop=add_self_edges) # (2, n_edges)
        receivers = edge_index[0, :]
        senders = edge_index[1, :]
        return receivers, senders

    def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
        # The model produces the output in normalized space so we apply inverse
        # normalization.
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats['std'].to(self._device)
        ) + acceleration_stats['mean'].to(self._device)

        # Use an Euler integrator to go from acceleration to position, assuming
        # a dt=1 corresponding to the size of the finite difference.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_position = most_recent_position + new_velocity  # * dt = 1
        return new_position

    def predict_positions(self, current_positions, n_particles_per_example, particle_types, friction, restitution):
        node_features, edge_index, e_features = self._build_graph_from_raw(current_positions, n_particles_per_example, particle_types, friction, restitution)
#        edge_index = edge_index.to(self._device)
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)
#        print('predicted',predicted_normalized_acceleration)
#        print('current',current_positions)
        next_position = self._decoder_postprocessor(predicted_normalized_acceleration, current_positions)
        return next_position

    def predict_accelerations(self, next_position, position_sequence_noise, position_sequence, n_particles_per_example, particle_types, friction, restitution, start_time):
        noisy_position_sequence = position_sequence + position_sequence_noise
#        print('noisy', noisy_position_sequence)
        node_features, edge_index, e_features = self._build_graph_from_raw(noisy_position_sequence, n_particles_per_example, particle_types, friction, restitution)

        start_time_4 = time.clock()
        #print(start_time_4 - start_time, "period4_")

#        edge_index = edge_index.to(self._device)
        #print('node_features',node_features)
        #print('edge_index',edge_index)
        #print('e_features',e_features)
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)
        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        #print('next',next_position_adjusted)
        target_normalized_acceleration = self._inverse_decoder_postprocessor(next_position_adjusted, noisy_position_sequence)
        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(self, next_position, position_sequence):
        """Inverse of `_decoder_postprocessor`."""
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity
#        print('acc',acceleration)
        acceleration_stats = self._normalization_stats["acceleration"]
#        print('acc_stats',acceleration_stats)
        normalized_acceleration = (acceleration - acceleration_stats['mean'].to(self._device)) / acceleration_stats['std'].to(self._device)
#        print('normal_acc',normalized_acceleration)
        return normalized_acceleration

    def save(self, state, LOG_DIR, path='model.pth'): #yujiang, comment: 'model.pth' is a default setting and will be used if nothing is passed into simulator.save()
        for infile in glob.glob(os.path.join(LOG_DIR, '*model.pth')):
            os.remove(infile) #yujiang, remove previous model.pth files stored, thus to only keep the latest model
        torch.save(state, path) #yujiang, changed from 'torch.save(self.state_dict(), path)'

    def load(self, path):
        self.load_state_dict(torch.load(path)['state_dict']) #yujiang, changed from 'self.load_state_dict(torch.load(path))'

def prepare_data_from_tfds(data_path=os.path.join(my_data_path, 'train.tfrecord'), is_rollout=False, batch_size=2):#yujiang, 22.01.06, previously 'data/train.tfrecord'
    import functools
    import tensorflow.compat.v1 as tf
    import tensorflow_datasets as tfds
    import reading_utils_friction_restitution as reading_utils
    import tree
    from tfrecord.torch.dataset import TFRecordDataset
    def prepare_inputs(tensor_dict):
        pos = tensor_dict['position']
        pos = tf.transpose(pos, perm=[1, 0, 2])
        target_position = pos[:, -1]
        tensor_dict['position'] = pos[:, :-1]
        num_particles = tf.shape(pos)[0]
        tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
        if 'step_context' in tensor_dict:
            tensor_dict['step_context'] = tensor_dict['step_context'][-2]
            tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
        return tensor_dict, target_position
    def batch_concat(dataset, batch_size):
        windowed_ds = dataset.window(batch_size)
        initial_state = tree.map_structure(lambda spec: tf.zeros(shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),dataset.element_spec)
        def reduce_window(initial_state, ds):
            return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))
        return windowed_ds.map(lambda *x: tree.map_structure(reduce_window, initial_state, x))
    def prepare_rollout_inputs(context, features):
        out_dict = {**context}
        pos = tf.transpose(features['position'], [1, 0, 2])
        target_position = pos[:, -1]
        out_dict['position'] = pos[:, :-1:sample_n_steps] # yujiang, 2022.02.03 changed
        out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
        if 'friction' in context:
            out_dict['friction'] = context['friction']
        if 'restitution' in context:
            out_dict['restitution'] = context['restitution']
        if 'width' in context:
            out_dict['width'] = context['width']
        if 'ratio' in context:
            out_dict['ratio'] = context['ratio']
        if 'step_context' in features:
            out_dict['step_context'] = features['step_context']
        out_dict['is_trajectory'] = tf.constant([True], tf.bool)
        return out_dict, target_position

    metadata = _read_metadata(my_data_path) #yujiang, 22.01.06, previously 'data/'
    ds = tf.data.TFRecordDataset([data_path])
    ds = ds.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))
    if is_rollout:
        ds = ds.map(prepare_rollout_inputs)
    else:
        split_with_window = functools.partial(
            reading_utils.split_trajectory, sample_n_steps = sample_n_steps,
            window_length=INPUT_SEQUENCE_LENGTH + 1) #yujiang, previously '6 + 1'
        ds = ds.flat_map(split_with_window)
        ds = ds.map(prepare_inputs)
        ds = ds.repeat()
        ds = ds.shuffle(512)
        ds = batch_concat(ds, batch_size)
    ds = tfds.as_numpy(ds)
#    for i in range(100): # clear screen
#        print() #yujiang, commented
    return ds


def eval_single_rollout(simulator, features, num_steps, device):
    initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]

    current_positions = initial_positions
    predictions = []
    for step in range(num_steps):
        next_position = simulator.predict_positions(
            current_positions,
            n_particles_per_example=features['n_particles_per_example'],
            particle_types=features['particle_type'],
            friction=features['friction'],
            restitution=features['restitution'],
            # width=features['width'],
            # ratio=features['ratio']
        ) # (n_nodes, 2)
        # Update kinematic particles from prescribed trajectory.
        kinematic_mask = (features['particle_type'] == kinematic_particle_id).clone().detach()
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, 2) #yujiang, 3d case
        next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
        predictions.append(next_position)
        current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)
    predictions = torch.stack(predictions) # (time, n_nodes, 2)
    ground_truth_positions = ground_truth_positions.permute(1,0,2)
    loss = (predictions - ground_truth_positions) ** 2 #yujiang, this evaluation loss is based on position differences for entire trajectory, 314 moments, e.g. shape: [314, 725, 2]
    output_dict = {
        'initial_positions': initial_positions.permute(1,0,2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'particle_types': features['particle_type'].cpu().numpy(),
    }
    loss = loss.mean() #yujiang
    return output_dict, loss #yujiang

def eval_rollout(ds, simulator, num_steps, num_eval_steps=1, save_results=False, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    eval_loss = []
    i = 0
    simulator.eval()
    with torch.no_grad():
        for example_i, (features, labels) in enumerate(ds):
            if i in [1]:
                features['position'] = torch.tensor(features['position']).to(device) # (n_nodes, 600, 2)
                features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
                features['particle_type'] = torch.tensor(features['particle_type']).to(device)
                features['friction'] = torch.tensor(features['friction']).to(device)
                features['restitution'] = torch.tensor(features['restitution']).to(device)
                # features['width'] = torch.tensor(features['width']).to(device)
                # features['ratio'] = torch.tensor(features['ratio']).to(device)
                labels = torch.tensor(labels).to(device)
                example_rollout, loss = eval_single_rollout(simulator, features, num_steps, device)
                example_rollout['metadata'] = metadata
                eval_loss.append(loss)
                if save_results:
                    print('No.'+str(i)+'validation trajectory')##yujiang, added
                    example_rollout['metadata'] = metadata
                    filename = f'optimized_rollout_{example_i}.pkl'
                    filename = os.path.join(my_rollouts_path_topic, filename)
                    with open(filename, 'wb') as f:
                        pickle.dump(example_rollout, f)
            i += 1
            if i >= num_eval_steps:
                break
    simulator.train()
    return example_rollout, loss #yujiang, originally 'return torch.stack(eval_loss).mean(0)'

def infer(simulator):
    start2 = datetime.now()
    ds = prepare_data_from_tfds(data_path=os.path.join(my_data_path, 'test.tfrecord'), is_rollout=True) #yujiang, 22.01.06, previously 'data/valid.tfrecord'
    print("Finished Training ! ", "Duration time:", datetime.now()-start2)
    return eval_rollout(ds, simulator, num_steps=num_steps, num_eval_steps=25, save_results=True, device=device) #yujiang, num_eval_steps=5 is added


class learnedsimulator_infer():
    def __init__(self, input_paramDict):
        self.start = datetime.now()  # yujiang, 22.01.06, added
        self.simulator = Simulator(
            particle_dimension=2,
            node_in=17,  # yujiang, originally 30 = 10+4+16, 2023.3.31: add 1 from 13 because friction is added
            edge_in=9,  # yujiang, originally 3, relative velocity is added into edge embedding, so we needs this
            # node_in=24, #yujiang, 3d case, basically 15+6=21
            # edge_in=4, #yujiang, 3d case
            latent_dim=128,
            num_message_passing_steps=8,  # yujiang, originally 10
            mlp_num_layers=2,
            mlp_hidden_dim=128,
            connectivity_radius=default_connectivity_radius,  # yujiang, 22.1.11, modified
            boundaries=np.array(metadata['bounds']),
            normalization_stats=normalization_stats,
            num_particle_types=9,  # yujiang, previously 9
            particle_type_embedding_size=3,
            input_paramDict = input_paramDict,
            device=device,
        )
    def infer2(self):
        if model_path is not None:
            self.simulator.load(model_path)
        if device == torch.device('cuda:0'):
            self.simulator.to(device)
        if is_train:  # yujiang, 22.01.06, added
            # train(self.simulator)
            pass
        else:
            outcome, loss = infer(self.simulator)
        print("Finished Training ! Duration time:", datetime.now() - self.start)  # yujiang, 22.01.06, added
        return outcome, loss