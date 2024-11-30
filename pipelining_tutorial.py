import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelArgs:
   dim: int = 512
   n_layers: int = 8
   n_heads: int = 8
   vocab_size: int = 10000

class Transformer(nn.Module):
   def __init__(self, model_args: ModelArgs):
      super().__init__()

      self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

      # Using a ModuleDict lets us delete layers witout affecting names,
      # ensuring checkpoints will correctly save and load.
      self.layers = torch.nn.ModuleDict()
      for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = nn.TransformerDecoderLayer(model_args.dim, model_args.n_heads)

      self.norm = nn.LayerNorm(model_args.dim)
      self.output = nn.Linear(model_args.dim, model_args.vocab_size)

   def forward(self, tokens: torch.Tensor):
      # Handling layers being 'None' at runtime enables easy pipeline splitting
      h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

      for layer in self.layers.values():
            h = layer(h, h)

      h = self.norm(h) if self.norm else h
      output = self.output(h).float() if self.output else h
      return output
   
import os
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
from datetime import datetime
global rank, device, pp_group, stage_index, num_stages, transfer_group
def init_distributed():
   global rank, device, pp_group, stage_index, num_stages, transfer_group
   rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
   dist.init_process_group(backend="nccl")
   pp_ranks = [0,1]
   # This group can be a sub-group in the N-D parallel case
   pp_group = dist.new_group(pp_ranks)
   transfer_ranks = [0, 1, 2, 3]
   transfer_group = dist.new_group(transfer_ranks)
   stage_index = rank if rank in pp_ranks else None
   num_stages = len(pp_ranks)
   # print()
   # print(f"rank {rank}, pp_group size: {len(pp_ranks)}, num_stages: {num_stages}")

def save_to_disk(tensor, path):
    """Utility function to save tensor to disk."""
    global rank
    torch.save(tensor, os.path.join(path, f"rank_{rank-2}_checkpoint.pth"))

def checkpoint_storage(stage=None, save_path="/projects/bdof/leatherman/pipeline_parallel_model_checkpoint"):
   """
   Handles sending a state_dict from a split model (PipelineStage) on src_rank
   to dst_rank and saving it to disk on the destination rank.
   """
   global rank, device
   if (rank == 0 or rank==1): # send 
      dst_rank = rank+2
      model_state_dict_cpu = stage.submod.state_dict()
      model_state_dict = {k: v.to(device) for k, v in model_state_dict_cpu.items()}
      current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
      # print(f"{current_time} Rank {rank}: model_state_dict: type {type(model_state_dict)}, length: {len(model_state_dict)}")
      # first, send a num
      tensor_to_send = torch.tensor(len(model_state_dict), dtype=torch.int64).to(device)
      current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
      print(f"0---{current_time} Rank {rank}: Send length of dict {len(model_state_dict)} with size {tensor_to_send.element_size()*tensor_to_send.numel()} to Rank {dst_rank}, device: {tensor_to_send.device}")
      dist.send(tensor_to_send, dst=dst_rank)
      # Serialize the state_dict as tensors
      for key, tensor in model_state_dict.items():
            # Send tensor key and data
            to_send_tensor1 = torch.tensor(len(key), dtype=torch.int64).to(device)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"1---{current_time} Rank {rank}: Sending tensor of size {to_send_tensor1.element_size()*to_send_tensor1.numel()} to Rank {dst_rank}")
            dist.send(to_send_tensor1, dst=dst_rank)

            to_send_tensor2 = torch.tensor(bytearray(key, 'utf-8')).to(device)
            print(f"{type(to_send_tensor2)}, dtype:, size: {len(to_send_tensor2)}")
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"2---{current_time} Rank {rank}: Sending tensor of size {to_send_tensor2.element_size()*to_send_tensor2.numel()} to Rank {dst_rank}")
            dist.send(to_send_tensor2, dst=dst_rank)

            ndim_tensor = torch.tensor(len(tensor.size()), dtype=torch.int64, device=device)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"3---{current_time} Rank {rank}: Sending tensor of size {ndim_tensor.element_size()*ndim_tensor.numel()} to Rank {dst_rank}")
            dist.send(ndim_tensor, dst=dst_rank)

            shape_tensor = torch.tensor(tensor.size(), dtype=torch.int64, device=device)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"4---{current_time} Rank {rank}: Sending tensor of size {shape_tensor.element_size()*shape_tensor.numel()} to Rank {dst_rank}")
            dist.send(shape_tensor, dst=dst_rank)
            # to_send_tensor3 = torch.tensor(len(tensor), dtype=torch.int64).to(device)
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # print(f"{current_time} Rank {rank}: Sending tensor of size {to_send_tensor3.element_size()*to_send_tensor3.numel()} to Rank {dst_rank}")
            # dist.send(to_send_tensor3, dst=dst_rank)

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"5---{current_time} Rank {rank}: Sending tensor of size {tensor.element_size()*tensor.numel()} to Rank {dst_rank}")
            dist.send(tensor, dst=dst_rank)

            print(f"Rank {rank} sent tensor '{key}' to Rank {dst_rank}")

   elif (rank == 2 or rank==3): # receive
      src_rank = rank-2
      # 1.first, get how many key_value pairs this rank is gonna receive:
      # Receive tensor key
      num_length_tensor = torch.empty(1, dtype=torch.int64).to(device)
      current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
      print(f"0---{current_time} Rank {rank}: receiving tensor of size {num_length_tensor.element_size()*num_length_tensor.numel()} from Rank {src_rank}")
      dist.recv(num_length_tensor, src=src_rank)

      number_of_dict_members = num_length_tensor.item()
      current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
      print(f"{current_time} rank: {rank},received num_length_tensor {number_of_dict_members}")
      received_state_dict = {}
      for i in range(number_of_dict_members):
         # Receive tensor key
         
         key_length = torch.empty(1, dtype=torch.int64).to(device)
         print(f"Rank {rank}: receiving key_length tensor of size {key_length.element_size()*key_length.numel()} from Rank {src_rank}")
         dist.recv(key_length, src=src_rank)
         print(f"the received key length is {key_length.item()}")

         key = torch.empty(key_length.item(), dtype=torch.uint8).to(device)
         print(f"Rank {rank}: receiving key tensor of size {key.element_size()*key.numel()} from Rank {src_rank}")
         dist.recv(key, src=src_rank)

         key = key.cpu().numpy().tobytes().decode('utf-8')
         print(f"Rank {rank}: received key is {key}")

         len_of_shape = torch.empty(1, dtype=torch.int64, device=device)
         dist.recv(len_of_shape, src=src_rank)
         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
         print(f"3---{current_time} Rank {rank}: Received tensor of size {len_of_shape.element_size()*len_of_shape.numel()} from Rank {src_rank}")

         ndim = len_of_shape.item()

         shape_tensor = torch.empty((ndim,), dtype=torch.int64, device=device)
         dist.recv(shape_tensor, src=src_rank)
         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
         print(f"4---{current_time} Rank {rank}: Received tensor of size {shape_tensor.element_size()*shape_tensor.numel()} from Rank {src_rank}")

         shape = tuple(shape_tensor.tolist())   

         # Receive tensor data
         tensor = torch.empty(shape, dtype=torch.float32, device=device)
         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
         dist.recv(tensor, src=src_rank)
         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
         print(f"5---{current_time} Rank {rank}: Received tensor of size {tensor.element_size()*tensor.numel()} from Rank {src_rank}")

         received_state_dict[key] = tensor.cpu()

      # Save the received state_dict to disk
      save_to_disk(received_state_dict, save_path)
      print(f"Rank {rank} saved the state_dict to {save_path}")

def manual_model_split(model, example_input_microbatch, model_args) -> PipelineStage:
   global pp_group, rank, device
   print(f"rank {rank}, pp_group size: {dist.get_world_size(pp_group)}")
   ckpt_path = os.path.join("/projects/bdof/leatherman/pipeline_parallel_model_checkpoint",f"rank_{rank}_checkpoint.pth")
   if stage_index == 0:
      # prepare the first stage model
      
      for i in range(4, 8):
            del model.layers[str(i)]
      model.norm = None
      model.output = None
      checkpoint = torch.load(ckpt_path, map_location=device)
      model.load_state_dict(checkpoint, strict=False)
      stage_input_microbatch = example_input_microbatch

   elif stage_index == 1:
      # prepare the second stage model
      for i in range(4):
            del model.layers[str(i)]
      model.tok_embeddings = None
      checkpoint = torch.load(ckpt_path, map_location=device)
      model.load_state_dict(checkpoint, strict=False)
      stage_input_microbatch = torch.randn(example_input_microbatch.shape[0], example_input_microbatch.shape[1], model_args.dim)

   stage = PipelineStage(
      model,
      stage_index,
      num_stages,
      device,
      group = pp_group,
      input_args=stage_input_microbatch,
   )
   return stage

if __name__ == "__main__":
   rank = int(os.getenv("LOCAL_RANK"))
   print(f"rank: {rank}, init distributed")
   init_distributed()
   num_microbatches = 4
   stage = None
   if rank==0 or rank==1:
      model_args = ModelArgs()
      model = Transformer(model_args)
      # model.load_state_dict("/projects/bdof/leatherman/pipeline_parallel_model_checkpoint")
      # Dummy data
      x = torch.ones(32, 500, dtype=torch.long)
      y = torch.randint(0, model_args.vocab_size, (32, 500), dtype=torch.long)
      example_input_microbatch = x.chunk(num_microbatches)[0]

      # Option 1: Manual model splitting
      # if rank==0 or rank==1:  
      stage = manual_model_split(model, example_input_microbatch, model_args)
      # Option 2: Tracer model splitting
      # stage = tracer_model_split(model, example_input_microbatch)
      x = x.to(device)
      y = y.to(device)

      def tokenwise_loss_fn(outputs, targets):
         loss_fn = nn.CrossEntropyLoss()
         outputs = outputs.view(-1, model_args.vocab_size)
         targets = targets.view(-1)
         return loss_fn(outputs, targets)

      schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)

      if rank == 0:
         schedule.step(x)
      elif rank == 1:
         losses = []
         output = schedule.step(target=y, losses=losses)
      # if stage_index == 0:
      #    torch.save(stage.model.state_dict(), "stage_0_checkpoint.pth")
      # elif stage_index == 1:
      #    torch.save(stage.model.state_dict(), "stage_1_checkpoint.pth")
   # here i want to have a barrier
   dist.barrier(group=transfer_group)
   checkpoint_storage(stage=stage)
   dist.destroy_process_group()