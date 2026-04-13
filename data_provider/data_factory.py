from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
import torch
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = False if flag == 'test' else True
    batch_size = args.batch_size
    freq = args.freq
    data_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        text_len=args.text_len,
    )
    if Data is Dataset_Custom:
        data_kwargs.update(
            use_scale_rag_cot=getattr(args, "use_scale_rag_cot", False),
            cot_model_name=getattr(args, "cot_model_name", "gpt2-medium"),
            cot_local_files_only=getattr(args, "cot_local_files_only", True),
            cot_max_new_tokens=getattr(args, "cot_max_new_tokens", 64),
            rag_stage1_topk=getattr(args, "rag_stage1_topk", 12),
            rag_stage2_topk=getattr(args, "rag_stage2_topk", 3),
            use_gpt2_rerank=getattr(args, "use_gpt2_rerank", False),
            gpt2_rerank_max_length=getattr(args, "gpt2_rerank_max_length", 512),
            use_longformer_rerank=getattr(args, "use_longformer_rerank", False),
            longformer_model_name=getattr(args, "longformer_model_name", "allenai/longformer-base-4096"),
            longformer_local_files_only=getattr(args, "longformer_local_files_only", True),
            longformer_max_length=getattr(args, "longformer_max_length", 2048),
            rag_long_topn=getattr(args, "rag_long_topn", 24),
            rag_cache_guidance=getattr(args, "rag_cache_guidance", True),
        )
    data_set = Data(**data_kwargs)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
