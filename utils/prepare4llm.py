from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import transformers


def _load_model_with_fallback(model_cls, model_name, config):
    attempts = [
        {"local_files_only": True},
        {"local_files_only": True, "use_safetensors": False},
        {"local_files_only": False, "use_safetensors": False},
        {"local_files_only": False},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            if kwargs["local_files_only"]:
                print(f"Loading {model_name} from local cache...")
            else:
                print(f"Local model files not found or invalid. Attempting to download {model_name}...")
            return model_cls.from_pretrained(model_name, config=config, **kwargs)
        except Exception as exc:
            last_error = exc
    raise last_error


def _load_tokenizer_with_fallback(tokenizer_cls, tokenizer_name):
    try:
        return tokenizer_cls.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception:
        print(f"Local tokenizer files not found or invalid. Attempting to download {tokenizer_name}...")
        return tokenizer_cls.from_pretrained(tokenizer_name, local_files_only=False)


def get_desc(domain, lookback_len, pred_len):
    
    description = {
        "Agriculture": ["Retail Broiler Composite", "month"],
        "Climate": ["Drought Level", "week"],
        "Economy": ["International Trade Balance", "month"],
        "Energy": ["Gasoline Prices", "week"],
        "Environment": ["Air Quality Index", "day"],
        "Health_US": ["Influenza Patients Proportion", "week"],
        "Security": ["Disaster and Emergency Grants", "month"],
        "SocialGood": ["Unemployment Rate", "month"],
        "Traffic": ["Travel Volume", "month"]
    }

    [OT, freq] = description[domain]

    desc = (f"Below is historical reporting information over the past {lookback_len} {freq}s concerning the {OT}. " 
        f"Based on these reports, predict the potential trends and anomalies of the {OT} for the next {pred_len} {freq}s.")
    return desc

def get_llm(llm_model:str, llm_layers:int=0):
    if llm_model == 'llama':
        # llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
        llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
        if llm_layers:
            llama_config.num_hidden_layers = llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        llm_model = _load_model_with_fallback(LlamaModel, 'huggyllama/llama-7b', llama_config)
        tokenizer = _load_tokenizer_with_fallback(LlamaTokenizer, 'huggyllama/llama-7b')
    elif llm_model == 'gpt2':
        gpt2_config = GPT2Config.from_pretrained('../llms/gpt2')
        if llm_layers:
            gpt2_config.num_hidden_layers = llm_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        llm_model = _load_model_with_fallback(GPT2Model, 'openai-community/gpt2', gpt2_config)
        tokenizer = _load_tokenizer_with_fallback(GPT2Tokenizer, 'openai-community/gpt2')
    elif llm_model == 'bert':
        bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
        if llm_layers:
            bert_config.num_hidden_layers = llm_layers
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True
        llm_model = _load_model_with_fallback(BertModel, 'google-bert/bert-base-uncased', bert_config)
        tokenizer = _load_tokenizer_with_fallback(BertTokenizer, 'google-bert/bert-base-uncased')
    else:
        raise Exception('LLM model is not defined')
    return llm_model, tokenizer
