import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import random

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils.utils import train, evaluate

def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")

parser = argparse.ArgumentParser(description="MCD-TSF")
parser.add_argument("--config", type=str, default="economy_36_18.yaml")
parser.add_argument("--datatype", type=str, default="multimodal")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=15)
parser.add_argument("--data", type=str, default="custom")
parser.add_argument("--embed", type=str, default="timeF")
parser.add_argument('--root_path', type=str, default='Time-MMD-main', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Economy/Economy.csv', help='data file')
parser.add_argument('--seq_len', type=int, default=36, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=18, help='prediction sequence length')
parser.add_argument('--text_len', type=int, default=36, help='context length in time series freq')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--freq', type=str, default='m', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--attn_drop', type=float, default=0.)
parser.add_argument('--init', type=str, default='None')
parser.add_argument('--valid_interval', type=int, default=1)
parser.add_argument('--time_weight', type=float, default=0.1)
parser.add_argument('--c_mask_prob', type=float, default=-1)
parser.add_argument('--beta_end', type=float, default=-1)
parser.add_argument('--lr', type=float, default=-1)
parser.add_argument('--save_attn', type=bool, default=False)
parser.add_argument('--save_token', type=bool, default=False)
parser.add_argument('--use_scale_rag_cot', action='store_true')
parser.add_argument('--cot_model_name', type=str, default='gpt2-medium')
parser.add_argument('--cot_local_files_only', type=str2bool, default=True)
parser.add_argument('--cot_max_new_tokens', type=int, default=64)
parser.add_argument('--rag_stage1_topk', type=int, default=12)
parser.add_argument('--rag_stage2_topk', type=int, default=3)
parser.add_argument('--use_longformer_rerank', action='store_true')
parser.add_argument('--longformer_model_name', type=str, default='allenai/longformer-base-4096')
parser.add_argument('--longformer_local_files_only', type=str2bool, default=True)
parser.add_argument('--longformer_max_length', type=int, default=2048)
parser.add_argument('--rag_long_topn', type=int, default=24)
parser.add_argument('--rag_cache_guidance', type=str2bool, default=True)
parser.add_argument('--use_scale_router', action='store_true')
parser.add_argument('--scale_router_aux_weight', type=float, default=0.05)
parser.add_argument('--scale_guidance_alpha', type=float, default=0.5)
parser.add_argument('--scale_guidance_min', type=float, default=0.5)
parser.add_argument('--scale_guidance_max', type=float, default=1.5)
parser.add_argument('--text_max_length', type=int, default=512)


args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.text_len == 0:
    args.text_len = args.seq_len

timestep_dim_dict = {
    'd': 3,
    'w': 2,
    'm': 1
}
context_dim_dict = {
    'bert': 768,
    'llama': 4096,
    'gpt2': 768
}
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)
if args.embed == 'timeF':
    if config["model"]["timestep_branch"] or config["model"]["timestep_emb_cat"]:
        config["model"]["timestep_dim"] = timestep_dim_dict[args.freq]
    else:
        config["model"]["timestep_dim"] = 0
else:
    config["model"]["timestep_dim"] = 4
config["model"]["context_dim"] = context_dim_dict[config["model"]["llm"]] if config["model"]["with_texts"] else 0

if args.datatype == 'electricity':
    target_dim = 370
    args.seq_len = 168
    args.pred_len = 24
else:
    target_dim = 1

config["model"]["is_unconditional"] = args.unconditional
config["model"]["lookback_len"] = args.seq_len
config["model"]["pred_len"] = args.pred_len
config["model"]["domain"] = args.data_path.split('/')[0]
config["model"]["text_len"] = args.text_len
config["model"]["save_attn"] = args.save_attn
config["model"]["save_token"] = args.save_token
config["model"]["text_max_length"] = args.text_max_length
config["model"]["use_scale_router"] = args.use_scale_router
config["model"]["scale_router_aux_weight"] = args.scale_router_aux_weight
config["model"]["scale_guidance_alpha"] = args.scale_guidance_alpha
config["model"]["scale_guidance_min"] = args.scale_guidance_min
config["model"]["scale_guidance_max"] = args.scale_guidance_max
config["diffusion"]["dropout"] = args.dropout
config["diffusion"]["attn_drop"] = args.attn_drop
config["diffusion"]["time_weight"] = args.time_weight
config["rag_cot"] = {
    "use_scale_rag_cot": args.use_scale_rag_cot,
    "cot_model_name": args.cot_model_name,
    "cot_local_files_only": args.cot_local_files_only,
    "cot_max_new_tokens": args.cot_max_new_tokens,
    "rag_stage1_topk": args.rag_stage1_topk,
    "rag_stage2_topk": args.rag_stage2_topk,
    "use_longformer_rerank": args.use_longformer_rerank,
    "longformer_model_name": args.longformer_model_name,
    "longformer_local_files_only": args.longformer_local_files_only,
    "longformer_max_length": args.longformer_max_length,
    "rag_long_topn": args.rag_long_topn,
    "rag_cache_guidance": args.rag_cache_guidance,
}

if args.use_scale_rag_cot:
    args.num_workers = 0

if args.c_mask_prob > 0:
    config["diffusion"]["c_mask_prob"] = args.c_mask_prob

if args.beta_end > 0:
    config["diffusion"]["beta_end"] = args.beta_end

if args.lr > 0:
    config["train"]["lr"] = args.lr

args.batch_size = config["train"]["batch_size"]

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.data_path.split('/')[0] + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config_results.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.datatype,
    device= args.device,
    batch_size=config["train"]["batch_size"],
    args=args
)

model = CSDI_Forecasting(config, args.device, target_dim, window_lens=[args.seq_len, args.pred_len]).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        valid_epoch_interval=args.valid_interval
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
model.target_dim = target_dim
if config["diffusion"]["cfg"]:
    best_mse = 10e10
    for guide_w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0]:
        mse = evaluate(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            window_lens=[args.seq_len, args.pred_len],
            guide_w=guide_w,
            save_attn=args.save_attn,
            save_token=args.save_token
        )
else:
    evaluate(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            window_lens=[args.seq_len, args.pred_len],
            save_attn=args.save_attn,
            save_token=args.save_token
        )
