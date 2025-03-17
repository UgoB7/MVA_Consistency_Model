"""import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from cm import dist_util, logger
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample, iterative_inpainting

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("Création du modèle et de la diffusion pour l'inpainting...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation="consistency" in args.training_mode,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    generator = get_generator(args.generator, args.num_samples, args.seed)
    model_kwargs = {}

    # Génération d'un échantillon avec karras_sample
    sample = karras_sample(
        diffusion,
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        steps=args.steps,
        model_kwargs=model_kwargs,
        device=dist_util.dev(),
        clip_denoised=args.clip_denoised,
        sampler=args.sampler,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        s_churn=args.s_churn,
        s_tmin=args.s_tmin,
        s_tmax=args.s_tmax,
        s_noise=args.s_noise,
        generator=generator,
        ts=tuple(float(t) for t in args.ts.split(",")) if args.ts else None,
    )

    # Ajustement du batch pour être un multiple de 7 (condition nécessaire pour iterative_inpainting)
    bs = sample.shape[0]
    if bs % 7 != 0:
        new_bs = bs - (bs % 7)
        sample = sample[:new_bs]
    else:
        new_bs = bs

    noise_x = generator.randn(new_bs, 3, args.image_size, args.image_size, device=dist_util.dev()) * args.sigma_max

    # Définition du débruiteur (denoiser) à passer à iterative_inpainting
    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if args.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    # Appel de iterative_inpainting qui retourne :
    #   - x_out : l'image inpaintée finale,
    #   - sample : l'image initiale avec le masque appliqué (zones masquées en noir),
    #   - mask : le masque utilisé.
    x_out, sample, mask = iterative_inpainting(
        distiller=denoiser,
        images=sample,
        x=noise_x,
        ts=tuple(float(t) for t in args.ts.split(",")) if args.ts else None,
        t_min=args.sigma_min,
        t_max=args.sigma_max,
        rho=diffusion.rho,
        steps=args.steps,
        generator=generator,
    )

    # Conversion de [-1, 1] vers [0, 255] pour x_out et sample
    x_out = ((x_out + 1) * 127.5).clamp(0, 255).to(th.uint8)
    x_out = x_out.permute(0, 2, 3, 1).contiguous()

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1).contiguous()

    # (Optionnel) Conversion du masque pour vérification : on suppose ici que c'est un masque binaire
    mask = (mask * 255).clamp(0, 255).to(th.uint8)
    mask = mask.permute(0, 2, 3, 1).contiguous()

    # Passage sur CPU et conversion en tableaux numpy pour sauvegarde
    sample_cpu = sample.cpu().numpy()
    x_out_cpu = x_out.cpu().numpy()
    mask_cpu = mask.cpu().numpy()

    # Création du dossier de sauvegarde
    save_dir = "/home/onyxia/work/consistency_models/scripts/results/inpainting"
    os.makedirs(save_dir, exist_ok=True)

    num_imgs = x_out_cpu.shape[0]
    for i in range(num_imgs):
        # Sauvegarde de l'image inpaintée seule
        inpainted_path = os.path.join(save_dir, f"inpainting_inpainted_{i:04d}.png")
        Image.fromarray(x_out_cpu[i]).save(inpainted_path)
        logger.log(f"Image inpaintée sauvegardée : {inpainted_path}")

        # Sauvegarde de l'image initiale avec masque en noir (à gauche)
        masked_path = os.path.join(save_dir, f"inpainting_masked_{i:04d}.png")
        Image.fromarray(sample_cpu[i]).save(masked_path)
        logger.log(f"Image initiale masquée sauvegardée : {masked_path}")

        # Sauvegarde de l'image du masque seule (optionnel)
        mask_path = os.path.join(save_dir, f"inpainting_mask_{i:04d}.png")
        Image.fromarray(mask_cpu[i]).save(mask_path)
        logger.log(f"Image du masque sauvegardée : {mask_path}")

        # Concaténation horizontale : image initiale avec masque à gauche, image inpaintée à droite
        pair = np.concatenate([sample_cpu[i], x_out_cpu[i]], axis=1)
        pair_path = os.path.join(save_dir, f"inpainting_pair_{i:04d}.png")
        Image.fromarray(pair).save(pair_path)
        logger.log(f"Image paire sauvegardée : {pair_path}")

    logger.log("Inpainting terminé.")

def create_argparser():
    defaults = dict(
        training_mode="consistency_distillation",
        generator="determ",
        clip_denoised=True,
        batch_size=32,
        steps=40,
        model_path="checkpoints/cd_bedroom256_lpips.pt",
        ts="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40",
        image_size=256,
        sigma_min=0.002,
        sigma_max=80.0,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        use_fp16=True,
        num_samples=32,
        seed=42,
        sampler="multistep"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
"""


#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample, iterative_inpainting

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # Détermine si l'on utilise la distillation (basée sur training_mode)
    distillation = "consistency" in args.training_mode

    logger.log("Création du modèle et du processus de diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("Génération d'un batch d'images avec karras_sample...")
    model_kwargs = {}
    generator = get_generator(args.generator, args.num_samples, args.seed)

    # Conversion du paramètre ts en tuple d'entiers
    ts = tuple(int(x) for x in args.ts.split(","))

    # Génération d'images initiales (valeurs dans [-1, 1])
    sample = karras_sample(
        diffusion,
        model,
        (args.batch_size, 3, args.image_size, args.image_size),
        steps=args.steps,
        model_kwargs=model_kwargs,
        device=dist_util.dev(),
        clip_denoised=args.clip_denoised,
        sampler=args.sampler,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        s_churn=args.s_churn,
        s_tmin=args.s_tmin,
        s_tmax=args.s_tmax,
        s_noise=args.s_noise,
        generator=generator,
        ts=ts,
    )

    # -------------------------------------------------------------
    # Affichage des images originales et avec masque appliqué
    # -------------------------------------------------------------
    # Conversion des images générées pour affichage (de [-1, 1] à [0, 255])
    sample_disp = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample_disp = sample_disp.permute(0, 2, 3, 1).cpu().numpy()

    # Création du masque à partir d'une image PIL
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt

    image_size = args.image_size
    # Créer une image blanche et dessiner une lettre (ici "S") en noir
    img = Image.new("RGB", (image_size, image_size), color="white")
    draw = ImageDraw.Draw(img)
    try:
        # Assurez-vous que la police "arial.ttf" est accessible, sinon modifiez le chemin.
        font = ImageFont.truetype("arial.ttf", 250)
    except IOError:
        font = ImageFont.load_default()
    draw.text((50, 0), "S", font=font, fill=(0, 0, 0))
    # Convertir en niveaux de gris et créer un masque binaire
    img_gray = img.convert("L")
    img_gray_np = np.array(img_gray)  # valeurs dans [0, 255]
    # Si pixel > 127, on considère la zone comme non masquée (1), sinon masquée (0)
    mask_np = (img_gray_np > 127).astype(np.float32)
    # Dupliquez sur 3 canaux pour correspondre à l'image
    mask_np = np.stack([mask_np]*3, axis=0)  # forme (3, H, W)
    mask_tensor = th.from_numpy(mask_np).to(sample.device)

    # Appliquer le masque à l'image : dans les zones masquées, on met -1 (correspondant à du noir)
    # Utilisation de la fonction de remplacement : x_mix = image * mask + (-1) * (1 - mask)
    mask_tensor = mask_tensor.unsqueeze(0)  # (1, 3, H, W) pour diffusion sur le batch
    masked_sample = sample * mask_tensor + (-th.ones_like(sample)) * (1 - mask_tensor)
    masked_sample_disp = ((masked_sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    masked_sample_disp = masked_sample_disp.permute(0, 2, 3, 1).cpu().numpy()

    # Affichage avec matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(sample_disp[0])
    axs[0].set_title("Image originale")
    axs[0].axis("off")
    axs[1].imshow(masked_sample_disp[0])
    axs[1].set_title("Image avec masque appliqué")
    axs[1].axis("off")
    plt.show()
    # -------------------------------------------------------------
    
    # Définition de la fonction denoiser (pour inpainting)
    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if args.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    #logger.log(f"Type de sample: {type(sample)} et shape: {sample.shape if isinstance(sample, th.Tensor) else 'N/A'}")
    #logger.log(f"Type de noise_x: {type(noise_x)} et shape: {noise_x.shape if isinstance(noise_x, th.Tensor) else 'N/A'}")


    if not isinstance(sample, th.Tensor):
        sample = th.tensor(sample, device=dist_util.dev())

    # Génération d'un bruit de même taille que les images (pour l'inpainting)
    noise_x = generator.randn(*sample.shape, device=dist_util.dev())

    #logger.log(f"Type de sample: {type(sample)} et shape: {sample.shape if isinstance(sample, th.Tensor) else 'N/A'}")
    #logger.log(f"Type de noise_x: {type(noise_x)} et shape: {noise_x.shape if isinstance(noise_x, th.Tensor) else 'N/A'}")

    logger.log("Application de l'inpainting itératif...")
    x_out, inpainted_sample = iterative_inpainting(
        distiller=denoiser,
        images=sample,
        x=noise_x,
        ts=ts,
        t_min=args.sigma_min,
        t_max=args.sigma_max,
        rho=args.rho,
        steps=args.steps,
        generator=generator,
    )

    # Conversion en uint8 pour sauvegarde/affichage
    x_out_disp = ((x_out + 1) * 127.5).clamp(0, 255).to(th.uint8)
    x_out_disp = x_out_disp.permute(0, 2, 3, 1).contiguous()
    inpainted_disp = ((inpainted_sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    inpainted_disp = inpainted_disp.permute(0, 2, 3, 1).contiguous()

    # Enregistrement avec matplotlib
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(sample_disp[0])
    axs[0].set_title("Image originale")
    axs[0].axis("off")
    axs[1].imshow(masked_sample_disp[0])
    axs[1].set_title("Image avec masque appliqué")
    axs[1].axis("off")
    fig.savefig("/home/onyxia/work/consistency_models/scripts/results/original_and_masked.png", bbox_inches="tight")
    plt.close(fig)

    # Sauvegarde (uniquement par le processus de rang 0)
    if dist.get_rank() == 0:
        out_dir = logger.get_dir()
        out_path = os.path.join(out_dir, f"inpainted_{args.batch_size}x{args.image_size}x{args.image_size}.npz")
        logger.log(f"Sauvegarde des images inpaintées dans {out_path}")
        np.savez(out_path,
                 original=sample_disp,
                 masked=masked_sample_disp,
                 inpainted=x_out_disp.cpu().numpy(),
                 inpainted_sample=inpainted_disp.cpu().numpy())
    
    dist.barrier()
    logger.log("Inpainting terminé.")

def create_argparser():
    defaults = dict(
        training_mode="consistency_distillation",
        generator="determ",
        clip_denoised=True,
        num_samples=7,
        batch_size=7,
        sampler="multistep",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40",
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
