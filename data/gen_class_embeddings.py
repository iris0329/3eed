from transformers import RobertaModel, RobertaTokenizerFast
import numpy as np

from data.model_util_scannet import ScannetDatasetConfig
import ipdb

st = ipdb.set_trace

config = ScannetDatasetConfig()
output_path = "data/class_embeddings3d.npy"

tokenizer = RobertaTokenizerFast.from_pretrained("/home/rongl/code/butd_detr_3d/data/roberta_base/FacebookAI--roberta-base.main.e2da8e2f811d1448a5b465c236feacd80ffbac7b")
text_encoder = RobertaModel.from_pretrained("/home/rongl/code/butd_detr_3d/data/roberta_base/FacebookAI--roberta-base.main.e2da8e2f811d1448a5b465c236feacd80ffbac7b")

object_list = [config.class2type[i] for i in range(len(config.class2type))]

tokenized = tokenizer.batch_encode_plus(object_list, padding="longest", return_tensors="pt")
encoded_text = text_encoder(**tokenized)
object_embeddings = (encoded_text.last_hidden_state * tokenized.attention_mask.unsqueeze(-1) / tokenized.attention_mask.sum(-1)[:, None, None]).sum(1)
np.save(output_path, object_embeddings.detach().numpy())
