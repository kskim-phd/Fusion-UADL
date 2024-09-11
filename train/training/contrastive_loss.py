import torch
import torch.distributed as dist
import diffdist.functional as distops # hbcho



def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''
    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')       행렬 곱 취해주는것. (mm을 해서 256,128 x 128,256 --> 256,256으로 나오게)
                                                #outputs.shape = 256,128 >> 256은 8n배치, 128은 simclr last_dim 256개가 각각 128개씩 값을 가지고 있음. 
    return sim_matrix   #sim_matrix.shape = 256,256


def NT_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''
    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')  256,256 diagonal 행렬 만듦 (단위행렬임 대각선 값이 전부 1)
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal 대각선 성분 전부 0값으로 만듦

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)   #256,1  (모든 행의 값 다 더함)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix    sim_matrix.shape = 256,256

    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B) #diag는 대각성분만 빼옴 따라서 반씩 나누고 (B가 128이므로) 반씩 대각성분만 빼와서 더하고
                                                                                      # 그럼 총 128개의 값이 나오고 그걸 다 더한 후 B'으로 나눠줌. (총 데이터 개수만큼)

    return loss


def Supervised_NT_xent(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    if multi_gpu:
        gather_t = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        labels = torch.cat(distops.all_gather(gather_t, labels))
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss

