from minference import get_support_models
from minference import MInferenceConfig 

supported_models = get_support_models()
supported_attn_types = MInferenceConfig.get_available_attn_types()
supported_kv_types = MInferenceConfig.get_available_kv_types()


print(f" >>> supported model: {supported_models}")
print(f" >>> supported attn-type: {supported_attn_types}")
print(f" >>> supported kv-type: {supported_kv_types}")



