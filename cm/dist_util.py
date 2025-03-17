"""
Helpers pour l'entraînement distribué (version simplifiée pour un seul GPU).
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Comme nous n'avons qu'un seul GPU, on fixe GPUS_PER_NODE à 1.
GPUS_PER_NODE = 1
SETUP_RETRY_COUNT = 3

def setup_dist():
    """
    Configure un groupe de processus distribué pour un seul GPU.
    Ici, nous fixons explicitement le rang à 0 et la taille du monde à 1.
    """
    # Si le groupe est déjà initialisé, on ne fait rien.
    if dist.is_initialized():
        return

    # Pour un seul GPU, on sélectionne toujours le GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Choix du backend : 'nccl' si CUDA est disponible, sinon 'gloo'.
    backend = "nccl" if th.cuda.is_available() else "gloo"

    # Configuration minimale des variables d'environnement pour torch.distributed.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_find_free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Initialisation du groupe de processus avec world_size=1 et rank=0.
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=0,
        world_size=1,
    )

def dev():
    """
    Retourne le device à utiliser (GPU si disponible, sinon CPU).
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Charge un fichier PyTorch.
    Comme nous ne distribuons pas le chargement entre plusieurs processus,
    nous pouvons simplement lire le fichier directement.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronisation des paramètres entre processus.
    Pour un seul GPU, cette fonction n'a pas besoin d'effectuer quoi que ce soit.
    """
    pass

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
