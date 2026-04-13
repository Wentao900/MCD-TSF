import os

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import transformers


def _resolve_local_model_path(model_name, aliases=()):
    candidates = [model_name, model_name.split("/")[-1], model_name.replace("/", "--")]
    candidates.extend(aliases)
    roots = [os.getcwd(), os.path.join(os.getcwd(), "llms"), os.path.join(os.getcwd(), "models")]
    for root in roots:
        for candidate in candidates:
            path = os.path.join(root, candidate)
            if os.path.isdir(path):
                return path
    raise FileNotFoundError(
        f"Local model files for {model_name} were not found under current directory, ./llms, or ./models. "
        f"Checked names: {', '.join(candidates)}"
    )


def _load_config(config_cls, model_name, aliases=()):
    model_path = _resolve_local_model_path(model_name, aliases)
    config = config_cls.from_pretrained(model_path, local_files_only=True)
    return config, model_path


def _load_model_local(model_cls, model_path, config):
    attempts = [
        {"local_files_only": True},
        {"local_files_only": True, "use_safetensors": False},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            print(f"Loading local model from {model_path}...")
            return model_cls.from_pretrained(model_path, config=config, **kwargs)
        except Exception as exc:
            last_error = exc
    raise last_error


def _load_tokenizer_local(tokenizer_cls, model_path):
    return tokenizer_cls.from_pretrained(model_path, local_files_only=True)


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
        llama_config, model_path = _load_config(LlamaConfig, 'huggyllama/llama-7b', aliases=('llama-7b', 'LLaMA-7B'))
        if llm_layers:
            llama_config.num_hidden_layers = llm_layers
        llama_config.output_attentions = True
        llama_config.output_hidden_states = True
        llm_model = _load_model_local(LlamaModel, model_path, llama_config)
        tokenizer = _load_tokenizer_local(LlamaTokenizer, model_path)
    elif llm_model == 'gpt2':
        gpt2_config, model_path = _load_config(GPT2Config, 'openai-community/gpt2', aliases=('gpt2',))
        if llm_layers:
            gpt2_config.num_hidden_layers = llm_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        llm_model = _load_model_local(GPT2Model, model_path, gpt2_config)
        tokenizer = _load_tokenizer_local(GPT2Tokenizer, model_path)
    elif llm_model == 'bert':
        bert_config, model_path = _load_config(BertConfig, 'google-bert/bert-base-uncased', aliases=('bert-base-uncased',))
        if llm_layers:
            bert_config.num_hidden_layers = llm_layers
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True
        llm_model = _load_model_local(BertModel, model_path, bert_config)
        tokenizer = _load_tokenizer_local(BertTokenizer, model_path)
    else:
        raise Exception('LLM model is not defined')
    return llm_model, tokenizer
