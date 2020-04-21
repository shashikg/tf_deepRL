from tf_deepRL.env.core import Env
from tf_deepRL.env.openai_gym import gymEnv

env_providers = ['OpenAI-gym']

def load_env(provider="OpenAI-gym", name=None):
    env = None
    if provider in env_providers:
        pass
    else:
        err_txt = ""
        for prov in env_providers: err_txt += prov + ", "
        raise KeyError("This environment provider is not supported. " + "Supported environment providers: " + err_txt[:-2])

    if provider == "OpenAI-gym":
        env = gymEnv(name)

    return env
